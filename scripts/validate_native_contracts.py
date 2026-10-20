"""Strict diagnostics: PS legality/native seeded trajectories and packed RL stepping.

No RNG injection, state repairs, ignored differences, or fallback actions.
Stops a trajectory's comparisons at its first state/control-flow divergence.
"""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import vgc_engine

FORMAT = 'gen9championsvgc2026regmc'
ROOT = Path(__file__).resolve().parents[1]
STATUS = dict(None_='', Sleep='slp', Freeze='frz', Paralysis='par', Burn='brn', Poison='psn', Toxic='tox')
STATUS['None'] = ''
BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']


class Oracle:
    def __init__(self):
        self.p = subprocess.Popen(['node', str(ROOT/'scripts/native_legality_oracle.cjs')],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def call(self, **payload):
        self.p.stdin.write(json.dumps(payload)+'\n')
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if 'error' in r:
            raise RuntimeError(r['error'])
        return r

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.p.terminate()
            self.p.wait(timeout=2)


def state_differences(b, ps, result='continue'):
    raw, obs = json.loads(b.to_json()), b.observe()
    vocab = vgc_engine.policy_vocab()
    differences = []
    def check(where, key, actual, expected):
        if actual != expected:
            differences.append(dict(at=where, field=key, native=actual, ps=expected))
    for side in range(2):
        s = raw[f'p{side+1}']
        for index, p in enumerate(ps['teams'][side]):
            m = s['team'][index]
            at = f'{side}:{index}'
            check(at, 'hp', m['current_hp'], p['hp'])
            check(at, 'maxhp', m['stats']['hp'], p['maxhp'])
            check(at, 'species', vocab['species'][m['species_id']], p['species'])
            check(at, 'item', vocab['items'][m['item_id']] if m['item_id'] < len(vocab['items']) else '', p['item'])
            ability=m.get('ability_override',65535)
            if ability==65535:ability=m['ability_id']
            check(at, 'ability', vocab['abilities'][ability] if ability < len(vocab['abilities']) else '', p['ability'])
            check(at, 'stats', {k:v for k,v in m['stats'].items() if k!='hp'}, p['stats'])
            # PS uses status='fnt' for a fainted mon; Rust stores faint in HP.
            check(at, 'status', 'fnt' if m['current_hp']==0 else STATUS.get(m['status'], m['status']),
                  'fnt' if p['hp']==0 else p['status'])
            check(at, 'boosts', m['boosts'], [p['boosts'][k] for k in BOOSTS])
            active = s['active'].index(index) if index in s['active'] else -1
            check(at, 'active', active, p['active'])
            check(at, 'pp', m['pp'][:len(p['moves'])], [x['pp'] for x in p['moves']])
            check(at, 'moves', [vocab['moves'][x] for x in m['moves'] if x<len(vocab['moves'])], [x['id'] for x in p['moves']])
    expected_end = 'continue' if not ps['ended'] else ('p1_win' if ps['winner']=='P1' else 'p2_win' if ps['winner']=='P2' else 'tie')
    weather={'none':'','sun':'sunnyday','rain':'raindance','sand':'sandstorm','snow':'snowscape','hail':'hail'}
    terrain={'none':'','electric':'electricterrain','grassy':'grassyterrain','misty':'mistyterrain','psychic':'psychicterrain'}
    check('field','weather',weather.get(obs['weather']['kind'],obs['weather']['kind']),ps['field']['weather'])
    check('field','terrain',terrain.get(obs['terrain']['kind'],obs['terrain']['kind']),ps['field']['terrain'])
    for native,showdown in [('trick_room','trickroom'),('gravity','gravity'),('magic_room','magicroom'),('wonder_room','wonderroom')]:
        check('field',native,raw[native+'_turns']>0,showdown in ps['field']['pseudo'])
    for side in range(2):
        conditions=raw[f'p{side+1}']['conditions']
        for native,showdown in [('reflect','reflect'),('light_screen','lightscreen'),('aurora_veil','auroraveil'),('tailwind','tailwind'),('safeguard','safeguard'),('mist','mist')]:
            check(f'side{side}',native,conditions[native+'_turns']>0,showdown in ps['sides'][side])
    check('battle', 'result', result, expected_end)
    return differences


def choose_joint(legal, rng):
    out=[]
    for side in range(2):
        chosen=[]; used=set(); mega=False
        for slot in range(2):
            choices=[tuple(c) for c in legal[side][slot]
                     if not (c[0]=='switch' and c[2] in used) and not (c[0]=='mega' and mega)]
            if not choices:
                raise ValueError(f'no compatible joint action: {side} {slot}')
            c=rng.choice(choices);chosen.append(c)
            if c[0]=='switch':used.add(c[2])
            mega |= c[0]=='mega'
        out.append(chosen)
    return out


def policy_action(c, side):
    kind,slot,arg,ts,tl=c
    if kind=='pass':return 0
    if kind=='switch':return arg+1
    target=0 if ts<0 else (-1 if ts==side else 1)*(tl+1)
    return 9+5*(0 if arg==4 else arg)+target+(20 if kind=='mega' else 0)


def policy_choices(b, side, actions):
    result=[]
    for slot,action in enumerate(actions):
        matches=[tuple(c) for c in b.legal_choices(side,slot) if policy_action(c,side)==int(action)]
        if len(matches)!=1:
            raise ValueError(f'policy action decode {side} {slot} {int(action)}: {matches}')
        result.append(matches[0])
    return result


def oracle_rollout(job, rpc, max_steps, rng):
    rng=random.Random(int(job['id'],16)+20261019055)
    init=rpc.call(op='init',format=FORMAT,seed=job['seed'],p1team=job['p1team'],p2team=job['p2team'])
    teams=[json.dumps(t) for t in init['sets']]
    b=vgc_engine.Battle.from_teams(*teams,format=FORMAT,stat_points=True,
        decision_phases=True,tera_allowed=False,ps_seed=','.join(map(str,job['seed'])))
    ps=init['state'];result='continue'; row=dict(id=job['id'],boundaries=0,legality_boundaries=0,legal_slots=0,
        legal_slot_mismatches=0,legality_differences=[],trace=[])
    for step in range(max_steps):
        dif=state_differences(b,ps,result)
        if dif:
            row.update(stop='state_divergence',differences=dif,step=step);break
        if result!='continue':
            row.update(stop='completed');break
        native=[[list(map(list,[tuple(c) for c in b.legal_choices(side,slot)])) for slot in range(2)] for side in range(2)]
        legal_reply=rpc.call(op='legal')
        assert legal_reply['seed']==ps['seed'], 'legality probes mutated original PS RNG'
        accepted=legal_reply['legal']
        row['legality_boundaries']+=1
        for side in range(2):
            for slot in range(2):
                n={tuple(c) for c in native[side][slot]};p={tuple(c) for c in accepted[side][slot]}
                row['legal_slots']+=1
                if n!=p:
                    row['legal_slot_mismatches']+=1
                    row['legality_differences'].append(dict(step=step,side=side,slot=slot,
                        rust_only=sorted(n-p),ps_only=sorted(p-n)))
        # Pick from Rust only; rejection remains a test failure, never repaired.
        choices=choose_joint(native,rng)
        seed_before=b.ps_seed
        result=b.step_move(*choices)
        reply=rpc.call(op='step',choices=choices);ps=reply['state']
        row['boundaries']+=1
        row['trace'].append(dict(step=step,choices=choices,result=result,
            native_seed_before=seed_before,native_seed_after=b.ps_seed,ps_seed_after=ps['seed']))
        if reply['errors']:
            row.update(stop='rust_action_rejected_by_ps',errors=reply['errors']);break
        if b.ps_seed!=ps['seed']:
            row.update(stop='rng_state_divergence',differences=state_differences(b,ps,result));break
    else:
        row.update(stop='step_cap')
    return row


def batch_rollout(jobs, rpc, max_steps, rng, checkpoint):
    from ai_vgc.nn.vgc_env import packed_policy_arrays, policy_vocab_tables
    from ai_vgc.nn.player import load_policy, sample_packed_batch
    import torch
    torch.set_num_threads(1)
    model=load_policy(checkpoint,'cpu')
    print('Loaded policy '+str(checkpoint),flush=True)
    results=[]
    for job in jobs:
        print('Starting packed rollout '+job['id'],flush=True)
        init=rpc.call(op='init',format=FORMAT,seed=job['seed'],p1team=job['p1team'],p2team=job['p2team'])
        teams=[json.dumps(t) for t in init['sets']]
        seed=int(job['id'],16)%(2**32)
        batch=vgc_engine.BattleBatch.from_teams([teams[0]],[teams[1]],format=FORMAT,
            seeds=[seed],stat_points=True,decision_phases=True,tera_allowed=False)
        one=vgc_engine.Battle.from_teams(*teams,format=FORMAT,seed=seed,
            stat_points=True,decision_phases=True,tera_allowed=False)
        vocab=policy_vocab_tables();batch.set_policy_vocab(*[vocab[k].tolist() for k in ['species','items','abilities','moves']])
        row=dict(id=job['id'],steps=0,selected_actions=0,trace=[])
        for step in range(max_steps):
            if batch.observe()[0]!=one.observe():
                row.update(stop='batch_vs_single_observation_mismatch',step=step);break
            packed=packed_policy_arrays(batch.policy_tensors(1700.,ots=False))
            _,actions,_,_=sample_packed_batch(model,packed,np.arange(2),greedy=False)
            choices=[policy_choices(one,s,actions[s]) for s in range(2)]
            masks=packed['mask']
            assert all(masks[s,sl,int(actions[s,sl])] for s in range(2) for sl in range(2))
            expected=one.step_move(*choices)
            codes=batch.step_policy_action_array(np.ascontiguousarray(actions[0:1],dtype=np.uint8),
                np.ascontiguousarray(actions[1:2],dtype=np.uint8),np.asarray([0],dtype=np.int64))
            actual=['unused','continue','p1_win','p2_win','tie'][int(np.frombuffer(codes,dtype=np.uint8)[0])]
            row['steps']+=1;row['selected_actions']+=4
            row['trace'].append(dict(actions=actions.tolist(),choices=choices,result=actual))
            if actual!=expected:
                row.update(stop='batch_vs_single_result_mismatch',expected=expected,actual=actual);break
            if batch.observe()[0]!=one.observe():
                row.update(stop='batch_vs_single_observation_mismatch',step=step);break
            if actual!='continue':row.update(stop='completed');break
        else:row.update(stop='step_cap')
        results.append(row)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['oracle','batch'],required=True)
    p.add_argument('--max-steps',type=int,default=120)
    p.add_argument('--checkpoint',default=str(ROOT/'data/models/mc-cts-rnad-v7f-MC378.pt'))
    args=p.parse_args();jobs=[json.loads(l) for l in args.jobs.read_text().splitlines() if l.strip()]
    rng=random.Random(20261019055);rpc=Oracle();rows=[]
    try:
        if args.mode=='batch':rows=batch_rollout(jobs,rpc,args.max_steps,rng,args.checkpoint)
        else:
            for job in jobs:
                try:r=oracle_rollout(job,rpc,args.max_steps,rng)
                except Exception as e:r=dict(id=job['id'],stop='harness_error',error=repr(e))
                rows.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ['trace','legality_differences']}),flush=True)
    finally:rpc.close()
    extension=next(Path(vgc_engine.__file__).parent.glob('*.so'))
    report=dict(mode=args.mode,extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),rows=rows,
        notes='No RNG outcomes injected. Oracle compares PS-seeded Python trajectories. Batch checks actual packed model sampling/default-RNG array stepping against single native instance, not directly PS.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(mode=args.mode,battles=len(rows),stops={s:sum(r['stop']==s for r in rows) for s in sorted({r['stop'] for r in rows})})))

if __name__=='__main__':main()
