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
const STAT = ['atk', 'def', 'spa', 'spd', 'spe'], STATUS = ['brn', 'par', 'psn', 'tox'];
const boostsOk = b => !b || Object.keys(b).every(k => STAT.includes(k));
const hasFn = m => Object.keys(m).some(k => typeof m[k] === 'function');
const secOk = m => !m.secondaries || (m.secondaries.length === 1 && Object.keys(m.secondaries[0]).every(k => ['chance', 'status', 'boosts', 'self'].includes(k))
	&& (!m.secondaries[0].status || STATUS.includes(m.secondaries[0].status)) && boostsOk(m.secondaries[0].boosts)
	&& (!m.secondaries[0].self || (Object.keys(m.secondaries[0].self).join() === 'boosts' && boostsOk(m.secondaries[0].self.boosts) && m.target === 'normal')));
const selfOk = m => !m.self || (Object.keys(m.self).join() === 'boosts' && boostsOk(m.self.boosts));
const common = m => !m.isNonstandard && !hasFn(m) && !m.isZ && !m.isMax && secOk(m) && !m.multihit && !m.drain && !m.recoil && !m.mindBlownRecoil
	&& !m.damage && !m.ohko && !m.selfdestruct && !m.forceSwitch && !m.selfSwitch && !m.volatileStatus && !m.sideCondition && !m.weather
	&& !m.terrain && !m.pseudoWeather && !m.heal && !m.flags.charge && !m.flags.recharge && !m.flags.futuremove && !m.breaksProtect
	&& !m.overrideOffensiveStat && !m.overrideDefensiveStat && !m.overrideOffensivePokemon && !m.willCrit && m.critRatio === 1 && !m.mindBlownRecoil
	&& !['Struggle', 'Future Sight', 'Doom Desire'].includes(m.name);
const damaging = m => common(m) && m.category !== 'Status' && m.basePower > 1 && !m.basePowerCallback && !m.boosts && !m.status
	&& selfOk(m) && ['normal', 'any', 'allAdjacentFoes', 'allAdjacent'].includes(m.target);
const statusMove = m => common(m) && m.category === 'Status' && (m.boosts || m.status) && boostsOk(m.boosts) && !m.self && !m.selfBoost
	&& (!m.status || STATUS.includes(m.status)) && ['normal', 'allAdjacentFoes', 'self'].includes(m.target);
const moves = dex.moves.all().filter(m => damaging(m) || statusMove(m));
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
	const parSkip = rnd(2) === 1;
	b.randomChance = (n, d) => (n === 1 && d === 8) ? parSkip : (n === 1 && d !== 100) ? false : true;
	b.makeChoices('team 1234', 'team 1234');
	const act = b.sides.flatMap(s => s.active);
	if (act.some(p => !p)) continue;
	// distinct speeds so ordering never needs the RNG tiebreak
	const status = act.map(() => [null, null, 'brn', 'par', 'psn', 'tox'][rnd(6)]);
	act.forEach((p, i) => { if (status[i]) p.setStatus(status[i]); });
	const startStatus = act.map(p => p.status || null); // immune types refuse the status
	const actions = act.map(() => ({slot: rnd(4), foe: rnd(2)}));
	const key = act.map((p, i) => dex.moves.get(teams[i >> 1][i & 1].moves[actions[i].slot]).priority * 1e6 + p.getStat('spe', false, false));
	if (new Set(key).size < 4) continue;
	const stats = act.map(p => ['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map(k => k === 'hp' ? p.maxhp : p.getStat(k, false, true)));
	const choice = side => [0, 1].map(j => {
		const a = actions[side * 2 + j], m = dex.moves.get(teams[side][j].moves[a.slot]);
		return `move ${a.slot + 1}` + (['normal', 'any'].includes(m.target) ? ` ${a.foe + 1}` : '');
	}).join(', ');
	try { b.makeChoices(choice(0), choice(1)); } catch (e) {
		console.error('BAD CHOICE', choice(0), '|', choice(1), teams.map(t => t.slice(0, 2).map(x => x.species + ':' + x.moves)), status, b.sides.map(sd => sd.activeRequest && JSON.stringify(sd.activeRequest.active.map(a => a.moves.map(mm => mm.id + (mm.disabled ? '(dis)' : ''))))));
		process.exit(1);
	}
	out.push({log: b.log.slice(b.log.findIndex(l => l === '|turn|1')).join('\n'), teams: teams.map(t => t.slice(0, 2)), actions, roll, stats, status: startStatus, parSkip, hp: act.map(p => p.hp),
		after: {status: act.map(p => p.status || null), boosts: act.map(p => STAT.map(k => p.boosts[k]))}});
}
console.log(JSON.stringify(out));
