// Multi-turn differential cases from Showdown: random 4v4 doubles games with moves, switches, KOs, forced
// replacements and field moves. RNG is forced (fixed damage roll per turn, no crits, accuracy and secondaries
// follow the roll). usage: node game_cases.js N_GAMES [MAX_TURNS] > data/gpusim/game_cases.json
const path = require('path');
const SD = path.resolve(__dirname, '../../pokemon-showdown-mc');
const {Battle} = require(path.join(SD, 'dist/sim/battle'));
const {Dex} = require(path.join(SD, 'dist/sim'));
const dex = Dex.mod('champions');
const N = +process.argv[2] || 200, MAX_TURNS = +process.argv[3] || 8;
let seed = +process.env.SEED || 4242;
const rnd = n => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed >> 8) % n; };
const species = dex.species.all().filter(s => !s.isNonstandard && !s.battleOnly && !s.requiredItem && !s.forme.match(/Mega|Gmax|Totem/));
const STAT = ['atk', 'def', 'spa', 'spd', 'spe'], STATUS = ['brn', 'par', 'psn', 'tox'];
const FIELD = ['sunnyday', 'raindance', 'sandstorm', 'snowscape', 'trickroom', 'tailwind', 'reflect', 'lightscreen', 'auroraveil',
	'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'];
const boostsOk = b => !b || Object.keys(b).every(k => STAT.includes(k));
const hasFn = m => Object.keys(m).some(k => typeof m[k] === 'function');
const secOk = m => !m.secondaries || (m.secondaries.length === 1 && Object.keys(m.secondaries[0]).every(k => ['chance', 'status', 'boosts', 'self'].includes(k))
	&& (!m.secondaries[0].status || STATUS.includes(m.secondaries[0].status)) && boostsOk(m.secondaries[0].boosts)
	&& (!m.secondaries[0].self || (Object.keys(m.secondaries[0].self).join() === 'boosts' && boostsOk(m.secondaries[0].self.boosts) && m.target === 'normal')));
const selfOk = m => !m.self || (Object.keys(m.self).join() === 'boosts' && boostsOk(m.self.boosts));
const common = m => !m.isNonstandard && !hasFn(m) && !m.isZ && !m.isMax && secOk(m) && !m.multihit && !m.drain && !m.recoil && !m.mindBlownRecoil
	&& !m.damage && !m.ohko && !m.selfdestruct && !m.forceSwitch && !m.selfSwitch && !m.volatileStatus && !m.sideCondition && !m.weather
	&& !m.terrain && !m.pseudoWeather && !m.heal && !m.flags.charge && !m.flags.recharge && !m.flags.futuremove && !m.breaksProtect
	&& !m.overrideOffensiveStat && !m.overrideDefensiveStat && !m.overrideOffensivePokemon && !m.willCrit && m.critRatio === 1
	&& !['Struggle', 'Future Sight', 'Doom Desire'].includes(m.name);
const damaging = m => common(m) && m.category !== 'Status' && m.basePower > 1 && !m.basePowerCallback && !m.boosts && !m.status
	&& selfOk(m) && ['normal', 'any', 'allAdjacentFoes', 'allAdjacent'].includes(m.target);
const statusMove = m => common(m) && m.category === 'Status' && (m.boosts || m.status) && boostsOk(m.boosts) && !m.self
	&& (!m.status || STATUS.includes(m.status)) && ['normal', 'allAdjacentFoes', 'self'].includes(m.target);
const fieldMove = m => FIELD.includes(m.id);
const SPECIAL = ['protect', 'detect', 'fakeout']; // implemented by id, not by the generic filters
const moves = dex.moves.all().filter(m => damaging(m) || statusMove(m) || fieldMove(m) || SPECIAL.includes(m.id));
const fieldMoves = moves.filter(fieldMove);
const natures = ['Adamant', 'Modest', 'Bold', 'Calm', 'Hardy', 'Jolly', 'Timid', 'Impish', 'Careful'];
const wcode = {sunnyday: 1, raindance: 2, sandstorm: 3, snowscape: 4, snow: 4}, tcode = {electricterrain: 1, grassyterrain: 2, mistyterrain: 3, psychicterrain: 4};

const snapshot = (b, names) => {
	const mons = b.sides.flatMap(s => s.pokemon);
	const id = p => +p.name.slice(1) + p.side.n * 4; // names are m0..m3 per side
	const hp = Array(8), status = Array(8), boosts = Array(8);
	for (const p of mons) { hp[id(p)] = p.hp; status[id(p)] = p.status || ''; boosts[id(p)] = STAT.map(k => p.boosts[k]); }
	const side = (s, c) => s.sideConditions[c] ? 1 : 0;
	return {hp, status, boosts, active: b.sides.flatMap(s => s.active.map(p => id(p))),
		weather: wcode[b.field.weather] || 0, terrain: tcode[b.field.terrain] || 0, tr: b.field.pseudoWeather.trickroom ? 1 : 0,
		tw: b.sides.map(s => side(s, 'tailwind')), reflect: b.sides.map(s => side(s, 'reflect')),
		ls: b.sides.map(s => side(s, 'lightscreen')), av: b.sides.map(s => side(s, 'auroraveil'))};
};

const out = [];
while (out.length < N) {
	const mk = k => {
		const s = species[rnd(species.length)];
		const ms = [0, 1, 2].map(() => moves[rnd(moves.length)].id);
		if (rnd(3) === 0) ms.push(fieldMoves[rnd(fieldMoves.length)].id); else ms.push(moves[rnd(moves.length)].id);
		return {species: s.name, name: 'm' + k, ability: 'Illuminate', nature: natures[rnd(natures.length)], level: 50, moves: ms,
			evs: {hp: rnd(33), atk: rnd(33), def: rnd(33), spa: rnd(33), spd: rnd(33), spe: rnd(33)}};
	};
	const teams = [0, 1].map(() => [0, 1, 2, 3].map(mk));
	let roll = 0, parSkip = false;
	let b;
	try {
		b = new Battle({formatid: 'gen9championsvgc2026regmc', seed: [1, 2, 3, 4],
			p1: {name: 'a', team: teams[0]}, p2: {name: 'b', team: teams[1]}});
	} catch (e) { continue; }
	b.random = () => roll;
	b.randomChance = (n, d) => (n === 1 && d === 8) ? parSkip : (n === 1 && d !== 100 && d !== 1) ? false : true;
	b.makeChoices('team 1234', 'team 1234');
	const game = {teams, stats: null, status0: null, turns: []};
	const mons = b.sides.flatMap(s => s.pokemon);
	const mid = p => +p.name.slice(1) + p.side.n * 4;
	game.stats = Array(8);
	for (const p of mons) game.stats[mid(p)] = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map(k => k === 'hp' ? p.maxhp : p.getStat(k, false, true));
	let ok = true;
	for (let turn = 0; turn < MAX_TURNS && !b.ended; turn++) {
		roll = rnd(16); parSkip = rnd(2) === 1;
		const actions = [], choices = ['', ''];
		for (let si = 0; si < 2; si++) {
			const side = b.sides[si], foes = b.sides[1 - si].active;
			const takenSwitch = new Set();
			const parts = [];
			for (let j = 0; j < 2; j++) {
				const p = side.active[j];
				if (p.fainted || !p.hp) { parts.push('pass'); actions.push([0, 0, 0]); continue; }
				const bench = side.pokemon.slice(2).filter(q => q.hp > 0 && !takenSwitch.has(q));
				if (bench.length && rnd(4) === 0) {
					const q = bench[rnd(bench.length)]; takenSwitch.add(q);
					parts.push(`switch ${side.pokemon.indexOf(q) + 1}`); actions.push([1, +q.name.slice(1), 0]); continue;
				}
				const req = side.activeRequest.active[j].moves, enabled = [0, 1, 2, 3].filter(x => !req[x].disabled);
				if (!enabled.length) { parts.push('move 1'); actions.push([0, 0, 0]); continue; } // Struggle
				const slot = enabled[rnd(enabled.length)], m = dex.moves.get(p.moves[slot]);
				const alive = [0, 1].filter(x => foes[x].hp > 0);
				const foe = alive.length ? alive[rnd(alive.length)] : 0;
				parts.push(`move ${slot + 1}` + (['normal', 'any'].includes(m.target) ? ` ${foe + 1}` : ''));
				actions.push([0, slot, foe]);
			}
			choices[si] = parts.join(', ');
		}
		const logStart = b.log.length;
		try { b.makeChoices(choices[0], choices[1]); } catch (e) { console.error('BAD CHOICE', choices, e.message, b.sides.map(sd => JSON.stringify(sd.activeRequest && sd.activeRequest.active.map(x => x.moves.map(mm => mm.move + (mm.disabled ? '(dis)' : '') + (mm.target||''))))), JSON.stringify(b.log.slice(-6)), b.sides.map(sd => sd.active.map(p => p.name + ':' + p.hp + ':' + p.moves.map(x => x + '/' + dex.moves.get(x).target).join(','))), b.sides.map(sd => sd.requestState)); process.exit(1); }
		const replace = [-1, -1, -1, -1];
		for (let guard = 0; guard < 3 && !b.ended && b.sides.some(s => s.requestState === 'switch'); guard++) {
			for (let si = 0; si < 2; si++) {
				const side = b.sides[si];
				if (side.requestState !== 'switch') continue;
				const need = side.active.map(p => p.fainted || !p.hp);
				const bench = side.pokemon.slice(2).filter(q => q.hp > 0);
				const parts = [];
				for (let j = 0; j < 2; j++) {
					if (need[j] && bench.length) { const q = bench.splice(rnd(bench.length), 1)[0]; parts.push(`switch ${side.pokemon.indexOf(q) + 1}`); replace[si * 2 + j] = +q.name.slice(1); }
					else parts.push('pass');
				}
				b.choose(si ? 'p2' : 'p1', parts.join(', '));
			}
		}
		const log = b.log.slice(logStart).join('\n');
		if (!b.ended && b.sides.some(s => s.requestState === 'switch')) { ok = false; break; }
		game.turns.push({actions, roll, parSkip, replace, log, after: snapshot(b)});
	}
	if (ok && game.turns.length) { game.status0 = null; out.push(game); }
}
console.log(JSON.stringify(out));
