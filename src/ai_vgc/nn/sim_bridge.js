// One-turn playouts for search (ai_vgc.nn.search). Reads JSON lines on stdin, writes one
// JSON line per request on stdout:
//
//   node src/ai_vgc/nn/sim_bridge.js pokemon-showdown
//   {"state": {...}, "pairs": [["move closecombat 1, move protect", "move 1 2, move 3"], ...],
//    "seeds": [1, 2]}
//   (with "dump": true instead: the rebuilt position, for checking)
//   -> {"results": [{"lines": [...], "winner": "us" | "them" | "tie" | null, "err": "..."}, ...]}
//
// `state` is the position as our bot sees it (built by search.py): format, turn, which side
// we are (`role`), field, and per side the Pokemon brought (actives first) with their sets and current
// HP, status, boosts and so on. The bridge starts a fresh battle with those teams, patches the
// state in, and plays every pair of choices once per seed (results are pair-major). `lines` are
// the turn's protocol lines as our side's client would receive them, so poke-env can parse them
// into a copy of the live battle.
'use strict';

const path = require('path');
const readline = require('readline');

const PS = path.resolve(process.argv[2] || 'pokemon-showdown', 'dist', 'sim');
const {Battle, extractChannelMessages} = require(PS + '/battle');
const {State} = require(PS + '/state');
const {PRNG} = require(PS + '/prng');

const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];

function patchMon(battle, p, m) {
  if (m.stats) {
    for (const s of ['atk', 'def', 'spa', 'spd', 'spe']) {
      if (m.stats[s]) p.storedStats[s] = p.baseStoredStats[s] = m.stats[s];
    }
  }
  if (m.maxhp) p.maxhp = p.baseMaxhp = m.maxhp;
  for (const v of m.volatiles || []) {
    try {
      if (v === 'leechseed') {
        const foe = p.side.foe.active.find(x => x && !x.fainted);
        if (foe) p.addVolatile(v, foe);
      } else if (v === 'perish1' || v === 'perish2' || v === 'perish3') {
        if (p.addVolatile('perishsong')) p.volatiles['perishsong'].duration = +v.slice(-1);
      } else {
        p.addVolatile(v);
      }
    } catch (e) {}
  }
  if (m.protect) {
    p.addVolatile('stall');
    if (p.volatiles['stall']) p.volatiles['stall'].counter = Math.pow(3, m.protect);
  }
  p.hp = m.fainted ? 0 : m.hpFrac !== undefined ? Math.max(1, Math.round(m.hpFrac * p.maxhp)) : m.hp;
  p.boosts = {};
  for (const b of BOOSTS) p.boosts[b] = (m.boosts && m.boosts[b]) || 0;
  p.item = m.item || '';
  p.itemState = battle.initEffectState({id: p.item, target: p});
  p.status = m.fainted ? 'fnt' : m.status || '';
  p.statusState = battle.initEffectState({id: p.status, target: p});
  if (p.status === 'slp') p.statusState.time = p.statusState.startTime = m.statusTurns || 2;
  if (p.status === 'tox') p.statusState.stage = m.statusTurns || 0;
  if (m.fainted) {
    p.fainted = true;
    p.clearVolatile(false);
  }
  p.activeTurns = p.activeMoveActions = m.firstTurn ? 0 : 1;
}

function build(st) {
  const battle = new Battle({formatid: st.format, seed: [1, 2, 3, 4]});
  const sides = {us: st.role, them: st.role === 'p1' ? 'p2' : 'p1'};
  for (const who of ['us', 'them']) {
    battle.setPlayer(sides[who], {name: who, team: st.sides[who].mons.map(m => m.set)});
  }
  for (const who of ['us', 'them']) {
    const n = st.sides[who].mons.length;
    battle.choose(sides[who], 'team ' + [...Array(n).keys()].map(i => i + 1).join(''));
  }
  const f = battle.field;
  f.weather = st.weather || '';
  f.weatherState = battle.initEffectState({id: f.weather, duration: st.weatherTurns || 0});
  f.terrain = st.terrain || '';
  f.terrainState = battle.initEffectState({id: f.terrain, duration: st.terrainTurns || 0});
  f.pseudoWeather = {};
  for (const [id, turns] of Object.entries(st.pseudo || {})) {
    f.pseudoWeather[id] = battle.initEffectState({id, duration: turns});
  }
  for (const who of ['us', 'them']) {
    const side = battle.getSide(sides[who]);
    const s = st.sides[who];
    side.sideConditions = {};
    for (const [id, turns] of Object.entries(s.conditions || {})) {
      side.sideConditions[id] = battle.initEffectState({id, target: side, duration: turns || undefined});
    }
    side.pokemon.forEach((p, i) => patchMon(battle, p, s.mons[i]));
    if (s.megaUsed) for (const p of side.pokemon) p.canMegaEvo = false;
    side.pokemonLeft = side.pokemon.filter(p => !p.fainted).length;
  }
  battle.turn = st.turn;
  battle.makeRequest('move');
  battle.log = [];
  return {json: JSON.stringify(State.serializeBattle(battle)), sides, battle};
}

function tryChoose(battle, side, choice) {
  if (battle.choose(side, choice)) return '';
  // A move that the sim thinks needs no target (or another one): drop the targets.
  const bare = choice.split(',').map(c => c.trim().replace(/^(move \S+)( -?\d)?/, '$1')).join(', ');
  battle.sides.find(s => s.id === side).clearChoice();
  if (battle.choose(side, bare)) return `retargeted ${choice}`;
  battle.sides.find(s => s.id === side).clearChoice();
  battle.choose(side, 'default');
  return `default for ${choice}`;
}

function dump(battle) {
  return battle.sides.map(side => side.pokemon.map(p => [
    p.name, p.species.id, p.hp, p.maxhp, p.status, p.item, p.ability,
    Object.entries(p.boosts).filter(([, v]) => v).map(([k, v]) => k + v).join(' '),
    Object.keys(p.volatiles).join(' '), p.isActive, p.activeMoveActions,
  ]).concat([[Object.keys(side.sideConditions).join(' ')]]))
    .concat([[battle.field.weather, battle.field.terrain, Object.keys(battle.field.pseudoWeather).join(' '), battle.turn]]);
}

// Team preview (Bo3 adaptation): {"start": {"format", "role", "them": [sets], "us": [[sets], ...]}}
// -> {"starts": [[lines], ...]}, the start-of-battle lines (leads sent out, switch-in abilities,
// up to turn 1) for each of our orders against theirs; both lists are leads first.
function start(req) {
  const st = req.start;
  const sides = {us: st.role, them: st.role === 'p1' ? 'p2' : 'p1'};
  const ch = +sides.us.slice(1);
  return {starts: st.us.map(ours => {
    const battle = new Battle({formatid: st.format, seed: [1, 2, 3, 4]});
    battle.send = () => {};
    const teams = {us: ours, them: st.them};
    for (const who of ['us', 'them']) battle.setPlayer(sides[who], {name: who, team: teams[who]});
    const n = battle.log.length;
    for (const who of ['us', 'them']) {
      battle.choose(sides[who], 'team ' + teams[who].map((_, i) => i + 1).join(''));
    }
    return extractChannelMessages(battle.log.slice(n).join('\n'), [ch])[ch];
  })};
}

function run(req) {
  if (req.start) return start(req);
  const {json, sides, battle} = build(req.state);
  if (req.dump) return {dump: dump(battle)};
  const ch = +sides.us.slice(1);
  const results = [];
  for (const [us, them] of req.pairs) {
    for (const seed of req.seeds) {
      const b = State.deserializeBattle(JSON.parse(json));
      b.prng = new PRNG([seed, 7, 11, 13]);
      b.send = () => {};
      let err = '';
      try {
        err = [tryChoose(b, sides.us, us), tryChoose(b, sides.them, them)].filter(Boolean).join('; ');
      } catch (e) {
        err = String(e && e.message || e);
      }
      const lines = extractChannelMessages(b.log.join('\n'), [ch])[ch];
      const winner = b.ended ? (b.winner === 'us' ? 'us' : b.winner === 'them' ? 'them' : 'tie') : null;
      results.push({lines, winner, err});
    }
  }
  return {results};
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', line => {
  let out;
  try {
    out = run(JSON.parse(line));
  } catch (e) {
    out = {error: String(e && e.stack || e)};
  }
  process.stdout.write(JSON.stringify(out) + '\n');
});
