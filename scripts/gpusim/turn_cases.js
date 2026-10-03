// One-turn differential cases from Showdown (forced RNG: no crits, every accuracy-100 move hits,
// damage roll fixed per case). usage: node turn_cases.js N > data/gpusim/turn_cases.json
const path = require('path');
const SD = path.resolve(__dirname, '../../pokemon-showdown-mc');
const {Battle} = require(path.join(SD, 'dist/sim/battle'));
const {Dex} = require(path.join(SD, 'dist/sim'));
const dex = Dex.mod('champions');
const N = +process.argv[2] || 1000;
let seed = 777;
const rnd = n => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed >> 8) % n; };
const species = dex.species.all().filter(s => !s.isNonstandard && !s.battleOnly && !s.requiredItem && !s.forme.match(/Mega|Gmax|Totem/));
const moves = dex.moves.all().filter(m => !m.isNonstandard && m.category !== 'Status' && m.basePower > 1 && (m.accuracy === true || m.accuracy === 100)
	&& !Object.keys(m).some(k => typeof m[k] === 'function') && !m.basePowerCallback && !m.willCrit && !m.selfBoost && !m.secondaries && !m.secondary && !m.self
	&& !m.multihit && !m.drain && !m.recoil && !m.mindBlownRecoil && !m.damage && !m.ohko && !m.selfdestruct && !m.forceSwitch
	&& !m.selfSwitch && !m.overrideOffensiveStat && !m.overrideDefensiveStat && !m.overrideOffensivePokemon
	&& !m.isZ && !m.isMax && !m.flags.charge && !m.flags.recharge && !m.volatileStatus && !m.status && !m.boosts && !m.sideCondition
	&& !m.breaksProtect && !m.flags.futuremove && ['normal', 'any', 'allAdjacentFoes', 'allAdjacent'].includes(m.target)
	&& !['Struggle', 'Future Sight', 'Doom Desire'].includes(m.name));
const natures = ['Adamant', 'Modest', 'Bold', 'Calm', 'Hardy', 'Jolly', 'Timid', 'Impish', 'Careful'];
const out = [];
while (out.length < N) {
	const mk = () => {
		const s = species[rnd(species.length)];
		return {species: s.name, ability: 'Pressure', nature: natures[rnd(natures.length)], level: 50,
			moves: [0, 1, 2, 3].map(() => moves[rnd(moves.length)].id),
			evs: {hp: rnd(33), atk: rnd(33), def: rnd(33), spa: rnd(33), spd: rnd(33), spe: rnd(33)}};
	};
	const teams = [[mk(), mk(), mk(), mk()], [mk(), mk(), mk(), mk()]];
	const roll = rnd(16);
	let b;
	try {
		b = new Battle({formatid: 'gen9championsvgc2026regmc', seed: [1, 2, 3, 4],
			p1: {name: 'a', team: teams[0]}, p2: {name: 'b', team: teams[1]}});
	} catch (e) { continue; }
	b.random = () => roll;
	b.randomChance = (n, d) => n >= d;
	b.makeChoices('team 1234', 'team 1234');
	const act = b.sides.flatMap(s => s.active);
	if (act.some(p => !p)) continue;
	// distinct speeds so ordering never needs the RNG tiebreak
	const actions = act.map(() => ({slot: rnd(4), foe: rnd(2)}));
	const key = act.map((p, i) => dex.moves.get(teams[i >> 1][i & 1].moves[actions[i].slot]).priority * 1e6 + p.getStat('spe', false, true));
	if (new Set(key).size < 4) continue;
	const stats = act.map(p => ['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map(k => k === 'hp' ? p.maxhp : p.getStat(k, false, true)));
	const choice = side => [0, 1].map(j => {
		const a = actions[side * 2 + j], m = dex.moves.get(teams[side][j].moves[a.slot]);
		return `move ${a.slot + 1}` + (['normal', 'any'].includes(m.target) ? ` ${a.foe + 1}` : '');
	}).join(', ');
	b.makeChoices(choice(0), choice(1));
	out.push({teams: teams.map(t => t.slice(0, 2)), actions, roll, stats, hp: act.map(p => p.hp)});
}
console.log(JSON.stringify(out));
