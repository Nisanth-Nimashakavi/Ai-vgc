"""Experimental model17 client using public Showdown observations and estimated sets."""
from __future__ import annotations
import argparse
import hashlib
import asyncio
import json
import os
import signal
import time
from pathlib import Path
import numpy as np
import torch
import vgc_engine
from poke_env import AccountConfiguration, ServerConfiguration, ShowdownServerConfiguration
from poke_env.battle import Pokemon, Move
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder
from ai_vgc.nn.model17 import Transformer17
from ai_vgc.nn.player import ChallengeNNPlayer, _public_login
from ai_vgc.nn.search import position, _mon_state
from ai_vgc.nn.vgc_bridge import VGCBridge
from ai_vgc.teams import RandomPoolTeambuilder


def load17(path, device):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    weights=ckpt.get('model_state_dict',ckpt.get('model',ckpt))
    model=Transformer17().to(device).eval()
    model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()})
    return model


def tensors(native, side, device):
    e=native.encode_17(side)
    c=torch.from_numpy(np.frombuffer(e['tok_cat'],dtype=np.uint32).reshape(1,17,4).copy()).long().to(device)
    n=torch.from_numpy(np.frombuffer(e['tok_num'],dtype=np.float32).reshape(1,17,16).copy()).to(device)
    c[:,4:8,1:3]=0
    return c,n


def public_native(battle, fmt, sets):
    # position only receives the client's public battle, never a full simulator state.
    state=position(battle,fmt,sets)
    if state is None:
        raise ValueError('Public position could not be reconstructed')
    native, us_p1=VGCBridge()._battle(state,seed=0)
    return native,0 if us_p1 else 1,state


def action_id(order,slot,battle,state):
    mon=order.order
    if isinstance(mon,Pokemon):
        names=[m['set']['name'] for m in state['sides']['us']['mons']]
        idx=names.index(mon.name)
        return 16 if idx==2 else 17
    if isinstance(mon,Move):
        active=battle.active_pokemon[slot]
        moves=list(active.moves)
        if mon.id not in moves: # Struggle has no learned move slot.
            return 3
        target=order.move_target
        ti=0 if target==1 else 1 if target==2 else 2 if target<0 else 3
        return moves.index(mon.id)*4+ti
    return 18


class Public17Player(ChallengeNNPlayer):
    def __init__(self, model, *, fmt, sets, decision_log=None, **kwargs):
        kwargs.pop('battle_format',None)
        super().__init__(model,greedy=True,rules=False,formats=[fmt],**kwargs)
        self.fmt,self.sets=fmt,sets
        self.decision_log=Path(decision_log) if decision_log else None
        self.inference_errors=0

    @torch.inference_mode()
    def preview_order(self,battle):
        ours=list(battle.team.values())
        theirs=sorted(battle.teampreview_opponent_team,key=lambda m:(m.species,m.name))
        states=[{'mons':[_mon_state(m,side==0,self.sets) for m in team]} for side,team in enumerate((ours,theirs))]
        native=vgc_engine.Battle.from_teams(*[VGCBridge._team(s) for s in states],format='gen9championsvgc2026regmc',stat_points=True)
        c,n=tensors(native,0,next(self.model.parameters()).device)
        plan=int(self.model(c,n)['preview_plan'].argmax().item())
        picks=list(vgc_engine.plan_to_picks(plan))
        if max(picks)>=len(ours):
            raise ValueError('Preview plan outside available team')
        for i,mon in enumerate(ours):
            mon._selected_in_teampreview=i in picks
        return '/team '+''.join(str(i+1) for i in picks)

    async def teampreview(self,battle):
        if self.timer:
            await self.ps_client.send_message('/timer on',battle.battle_tag)
        order=self.preview_order(battle)
        self.last_preview[battle.battle_tag]=order
        return order

    @torch.inference_mode()
    def public_order(self,battle):
        native,side,state=public_native(battle,self.fmt,self.sets)
        legal=DoubleBattleOrder.join_orders(*battle.valid_orders)
        if not legal: raise ValueError('No request-legal joint order')
        pairs=[(action_id(o.first_order,0,battle,state),action_id(o.second_order,1,battle,state)) for o in legal]
        device=next(self.model.parameters()).device
        c,n=tensors(native,side,device)
        a_logits=self.model(c,n)['policy_a'][0].float()
        ids_a=sorted({a for a,b in pairs})
        ma=torch.full_like(a_logits,float('-inf')); ma[ids_a]=a_logits[ids_a]
        logpa=ma.log_softmax(-1)
        out=self.model(c.expand(len(ids_a),-1,-1),n.expand(len(ids_a),-1,-1),action_a=torch.tensor(ids_a,device=device))
        scores={}
        for i,a in enumerate(ids_a):
            logits=out['policy_b'][i].float()
            ids_b=sorted({b for aa,b in pairs if aa==a})
            masked=torch.full_like(logits,float('-inf')); masked[ids_b]=logits[ids_b]
            for b in ids_b: scores[a,b]=float(logpa[a]+masked.log_softmax(-1)[b])
        idx=max(range(len(legal)),key=lambda i:(scores[pairs[i]], bool(legal[i].first_order.mega or legal[i].second_order.mega)))
        return legal[idx]

    async def choose_move(self,battle):
        start=time.time()
        try:
            order=self.public_order(battle)
        except Exception as exc:
            self.inference_errors+=1
            # Keep the game running and explicitly report every fallback.
            self.logger.exception('Public model17 inference failed')
            orders=DoubleBattleOrder.join_orders(*battle.valid_orders)
            if not orders: raise
            order=orders[0]
        if self.decision_log:
            self.decision_log.parent.mkdir(parents=True,exist_ok=True)
            with self.decision_log.open('a') as handle:
                handle.write(json.dumps(dict(tag=battle.battle_tag,turn=battle.turn,order=order.message,
                    elapsed=time.time()-start,inference_errors=self.inference_errors,observation='public reconstructed estimates'))+'\n')
        return order


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--username',default='nimnimbot')
    p.add_argument('--games',type=int,default=10)
    p.add_argument('--teams',default='data/teams/reg_mc')
    p.add_argument('--only',nargs='+',default=['MC378'])
    p.add_argument('--format',default='gen9championsvgc2026regmc')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True)
    p.add_argument('--local',action='store_true')
    p.add_argument('--port',type=int,default=18791)
    p.add_argument('--opponent',default='data/models/mc-cts-rnad-v6-native-pfsp-v2.pt')
    args=p.parse_args()
    torch.set_num_threads(1)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    (root/'manifest.json').write_text(json.dumps(dict(model=args.model,model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),games=args.games,format=args.format,teams=args.only,observation='public reconstructed estimates; raw model17, no native search'),indent=2))
    proc=None
    if args.local:
        from ai_vgc.showdown import ensure_server
        proc=ensure_server(args.port,'pokemon-showdown-mc')
    config=ServerConfiguration(f'ws://localhost:{args.port}/showdown/websocket','https://play.pokemonshowdown.com/action.php?') if args.local else ShowdownServerConfiguration
    bot=Public17Player(load17(args.model,args.device),fmt=args.format,sets=args.teams,
        decision_log=root/'decisions.jsonl',team=RandomPoolTeambuilder(Path(args.teams),names=args.only),
        account_configuration=AccountConfiguration(args.username,os.environ.get('PS_PASSWORD') if not args.local else None),
        battle_format=args.format,server_configuration=config,accept_open_team_sheet=False,max_concurrent_battles=1,
        start_listening=args.local,ping_interval=20,ping_timeout=60)
    bot.formats=[args.format]; bot.model_name=Path(args.model).stem
    bot.save_dir=root; bot.watch_url='https://play.pokemonshowdown.com/' if not args.local else f'http://localhost:{args.port}/'
    bot.timer=not args.local
    signal.signal(signal.SIGINT,lambda *_:setattr(bot,'stop_after_match',True))
    print(f'Experimental public-observation model17: {args.games} games; model {args.model}',flush=True)
    if args.local:
        from ai_vgc.nn.player import NNPlayer
        other=NNPlayer(args.opponent,greedy=True,team=RandomPoolTeambuilder(Path(args.teams),names=args.only),
            account_configuration=AccountConfiguration('model17-baseline',None),battle_format=args.format,
            server_configuration=config,max_concurrent_battles=1)
        asyncio.run(bot.battle_against(other,n_battles=args.games))
    else:
        _public_login(bot,os.environ.get('PS_PASSWORD'))
        asyncio.run(bot.ladder(args.games))
    result=dict(wins=bot.n_won_battles,games=bot.n_finished_battles,inference_errors=bot.inference_errors,
                observation='public reconstructed estimates; differs from privileged native training')
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if proc is not None: proc.terminate()

if __name__=='__main__': main()
