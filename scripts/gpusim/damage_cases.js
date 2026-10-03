// Generate damage test cases from Showdown: usage: node damage_cases.js N > cases.json
const path = require('path');
const SD = path.resolve(__dirname, '../../pokemon-showdown-mc');
const {Battle} = require(path.join(SD, 'dist/sim/battle'));
const {Dex} = require(path.join(SD, 'dist/sim'));
const dex = Dex.mod('champions');
const N = +process.argv[2] || 2000;
let seed = 12345;
const rnd = n => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
const species = dex.species.all().filter(s => !s.battleOnly && !s.forme.match(/Mega|Gmax/));
const moves = dex.moves.all().filter(m => m.basePower > 1 && !m.basePowerCallback && !m.damageCallback && !m.damage && !m.multihit && !m.isZ && !m.isMax
	&& !m.overrideOffensiveStat && !m.overrideDefensiveStat && !m.overrideOffensivePokemon && !m.onModifyMove && !m.onBasePower && !['Struggle', 'Future Sight', 'Doom Desire'].includes(m.name) && m.category !== 'Status');
const out = [];
const natures = ['Adamant', 'Modest', 'Bold', 'Calm', 'Hardy', 'Jolly'];
while (out.length < N) {
	const sa = species[rnd(species.length)], sd = species[rnd(species.length)], mv = moves[rnd(moves.length)];
	const mk = (s, extra) => ({species: s.name, ability: 'Pressure'  /* neutral: ability effects are tested separately */, moves: [mv.id], nature: natures[rnd(6)],
		evs: {hp: rnd(33), atk: rnd(33), def: rnd(33), spa: rnd(33), spd: rnd(33), spe: rnd(33)}, level: 50, ...extra});
	let b;
	try {
		b = new Battle({formatid: 'gen9championsvgc2026regmc', seed: [1, 2, 3, 4],
			p1: {name: 'a', team: [mk(sa), mk(sa)]}, p2: {name: 'b', team: [mk(sd), mk(sd)]}});
	} catch (e) { if (!global.__e) { global.__e = 1; console.error(e.message.slice(0, 300)); } continue; }
	b.makeChoices('team 12', 'team 12');
	const A = b.sides[0].active[0], D = b.sides[1].active[0];
	if (!A || !D) continue;
	const roll = rnd(16), spread = rnd(2) === 1, burned = rnd(4) === 0 && mv.category === 'Physical';
	b.random = () => roll;
	const am = dex.getActiveMove(mv.id);
	am.willCrit = false; am.spreadHit = spread;
	if (burned) A.status = 'brn';
	const typeMod = D.runEffectiveness(am);
	if (!D.runImmunity(am.type)) continue; // immunity lives in the hit pipeline, tested there
	const immune = false;
	const dmg = b.actions.getDamage(A, D, am, true);
	if (dmg === false || dmg === undefined || dmg === null) continue;
	const phys = mv.category === 'Physical';
	out.push({move: mv.id, level: 50, power: am.basePower, atk: A.getStat(phys ? 'atk' : 'spa', false, true),
		dfn: D.getStat(phys ? 'def' : 'spd', false, true), roll, spread, burned, crit: false,
		stab: A.hasType(am.type) ? 6144 : 4096, type_exp: immune ? -99 : typeMod,
		abilityA: A.ability, abilityD: D.ability, expected: dmg});
}
console.log(JSON.stringify(out));
