/* Local JSON-lines oracle: no RNG forcing or outcomes imported from Rust. */
'use strict';
const readline = require('readline');
const {Battle, Teams} = require('../pokemon-showdown-mc/dist/sim');
let b, original;
const copy = x => JSON.parse(JSON.stringify(x));
function snapshot() {
  return {
    turn:b.turn, ended:b.ended, winner:b.winner, seed:b.prng.getSeed(),
    request:b.requestState,
    teams:b.sides.map((s,side) => original[side].map(name => {
      const p=s.pokemon.find(p=>p.name===name);
      return {name, species:p.species.id, hp:p.hp, maxhp:p.maxhp,
        stats:p.storedStats, status:p.status, item:p.item, ability:p.ability,
        boosts:p.boosts, active:s.active.indexOf(p),
        moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp})),
        volatiles:Object.keys(p.volatiles).sort()};
    })),
    field:{weather:b.field.weather,terrain:b.field.terrain,
      pseudo:Object.keys(b.field.pseudoWeather).sort()},
    sides:b.sides.map(s=>Object.keys(s.sideConditions).sort()),
  };
}
function command(c, side, battle=b) {
  const [kind,slot,arg,ts,tl]=c;
  if(kind==='pass')return 'pass';
  if(kind==='switch')return `switch ${battle.sides[side].pokemon.findIndex(p=>p.name===original[side][arg])+1}`;
  const p=battle.sides[side].active[slot];
  const name=p.volatiles.mustrecharge?'recharge':arg===4?'struggle':p.moveSlots[arg]?.id;
  return `move ${name}${ts<0?'':` ${ts===side?'-':''}${tl+1}`}${kind==='mega'?' mega':''}`;
}
function legal(side, slot, serialized) {
  const probe=Battle.fromJSON(serialized);
  const source=probe.sides[side], p=source.active[slot], req=p.getMoveRequestData();
  // No submission is possible on a waiting side. Rust represents this with
  // pass padding; do not test an actual PS "pass" command in that phase.
  if(!source.activeRequest || source.activeRequest.wait)return [['pass',slot,0,-1,-1]];
  // chooseSwitch auto-skips non-switching slots. Without this guard, a probe
  // labelled slot 0 could actually validate a switch for slot 1.
  if(source.requestState==='switch' && !p.switchFlag)return [['pass',slot,0,-1,-1]];
  const candidates=[['pass',slot,0,-1,-1]];
  for(let i=0;i<original[side].length;i++)candidates.push(['switch',slot,i,-1,-1]);
  const virtual=req.moves.length===1 && ['struggle','recharge'].includes(req.moves[0].id);
  for(let i=0;i<req.moves.length;i++) {
    let arg=virtual&&req.moves[i].id==='struggle'?4:p.moveSlots.findIndex(m=>m.id===req.moves[i].id);
    if(virtual&&req.moves[i].id==='recharge')arg=Math.max(0,p.moveSlots.findIndex(m=>m.id===p.lastMove?.id));
    for(const kind of ['move','mega'])for(const [ts,tl] of [[-1,-1],[0,0],[0,1],[1,0],[1,1]])candidates.push([kind,slot,arg,ts,tl]);
  }
  const accepted=[];
  for(const action of candidates) {
    const c=Battle.fromJSON(serialized), s=c.sides[side], mon=s.active[slot];
    s.clearChoice();
    s.choice.actions=Array.from({length:slot},()=>({choice:'pass'}));
    let ok=false;
    if(action[0]==='pass')ok=!!s.choosePass();
    else if(s.activeRequest?.wait)ok=false;
    else if(action[0]==='switch') {
      const i=s.pokemon.findIndex(p=>p.name===original[side][action[2]]);
      ok=!!s.chooseSwitch(String(i+1));
    } else if(!mon.fainted && !mon.volatiles.commanding) {
      const virtualName=mon.volatiles.mustrecharge?'recharge':action[2]===4?'struggle':mon.moveSlots[action[2]]?.id;
      const loc=action[3]<0?0:(action[3]===side?-1:1)*(action[4]+1);
      ok=!!s.chooseMove(virtualName,loc,action[0]==='mega'?'mega':'');
    }
    if(ok && (action[0]!=='mega' || s.choice.mega))accepted.push(action);
  }
  return accepted;
}
function handle(x) {
  if(x.op==='init') {
    const sets=[x.p1team,x.p2team].map((t,s)=>Teams.import(t).map((p,i)=>({...p,name:`p${s+1}m${i}`})));
    original=sets.map(t=>t.map(p=>p.name));
    b=new Battle({formatid:x.format,seed:x.seed,p1:{name:'P1',team:sets[0]},p2:{name:'P2',team:sets[1]}});
    b.choose('p1','team 1234');b.choose('p2','team 1234');
    return {state:snapshot(),sets};
  }
  if(x.op==='legal') {
    const serialized=JSON.stringify(b.toJSON());
    const result=[0,1].map(s=>[0,1].map(slot=>legal(s,slot,serialized)));
    if(JSON.stringify(b.toJSON())!==serialized)throw Error('legality probes mutated original battle');
    return {legal:result,seed:b.prng.getSeed(),original_state_unchanged:true};
  }
  if(x.op==='step') {
    const errors=[];
    // A one-sided replacement can immediately advance to a NEW request. Never
    // answer that new request with the other side's old padding choices.
    const canAct=b.sides.map(s=>!!s.activeRequest && !s.activeRequest.wait);
    const commands=x.choices.map((cs,side)=>cs.map(c=>command(c,side)).join(', '));
    for(let side=0;side<2;side++) {
      if(!canAct[side])continue;
      const cmd=commands[side];
      if(!b.choose(`p${side+1}`,cmd))errors.push({side,cmd,error:b.sides[side].choice.error});
    }
    return {state:snapshot(),errors};
  }
  throw Error('unknown operation');
}
readline.createInterface({input:process.stdin}).on('line',line=>{
  try {process.stdout.write(JSON.stringify(handle(JSON.parse(line)))+'\n');}
  catch(e){process.stdout.write(JSON.stringify({error:e.stack})+'\n');}
});
