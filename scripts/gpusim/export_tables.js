// Dump the Reg M-C data from the local Showdown fork as JSON for the GPU sim.
// usage: node scripts/gpusim/export_tables.js > data/gpusim/tables.json
const path = require('path');
const SD = path.resolve(__dirname, '../../pokemon-showdown-mc');
const {Dex} = require(path.join(SD, 'dist/sim'));
const FORMAT = 'gen9championsvgc2026regmc';
const dex = Dex.mod('champions');
const fmt = Dex.formats.get(FORMAT);
const rules = Dex.formats.getRuleTable(fmt);

const ok = (kind, o) => o.exists && !o.isNonstandard && !rules.check(`${kind}:${o.id}`);
const learnable = new Set();
for (const s of dex.species.all()) {
	const ls = dex.species.getLearnsetData(s.id);
	if (ls && ls.learnset) for (const m of Object.keys(ls.learnset)) learnable.add(m);
}
const species = dex.species.all().filter(s => ok('pokemon', s) && !s.battleOnly || (ok('pokemon', s) && s.battleOnly));
const moves = dex.moves.all().filter(m => ok('move', m) && learnable.has(m.id) && !m.isZ && !m.isMax);
const abilities = dex.abilities.all().filter(a => ok('ability', a));
const items = dex.items.all().filter(i => ok('item', i));

const HANDLER = /^on[A-Z]/;
const handlers = o => Object.keys(o).filter(k => HANDLER.test(k) && typeof o[k] === 'function');
const types = dex.types.names();
const chart = types.map(a => types.map(d => {
	const e = dex.types.get(d).damageTaken[a];
	return e === 1 ? 2 : e === 2 ? 0.5 : e === 3 ? 0 : 1;
}));

const pick = (o, ks) => Object.fromEntries(ks.map(k => [k, o[k]]));
console.log(JSON.stringify({
	format: FORMAT, types, typeChart: chart,
	species: species.map(s => ({...pick(s, ['id', 'name', 'num', 'types', 'baseStats', 'abilities', 'weightkg', 'baseSpecies', 'forme', 'requiredItem', 'battleOnly', 'changesFrom']), }) ),
	moves: moves.map(m => ({...pick(m, ['id', 'name', 'num', 'type', 'category', 'basePower', 'accuracy', 'pp', 'priority', 'target', 'flags', 'secondaries', 'boosts', 'self', 'status', 'volatileStatus', 'sideCondition', 'weather', 'terrain', 'pseudoWeather', 'drain', 'recoil', 'heal', 'multihit', 'critRatio', 'ohko', 'selfdestruct', 'forceSwitch', 'selfSwitch', 'damage', 'overrideOffensiveStat', 'overrideDefensiveStat', 'overrideOffensivePokemon', 'ignoreAbility', 'breaksProtect', 'isProtect', 'ignoreImmunity', 'selfBoost', 'willCrit', 'tracksTarget', 'noPPBoosts', 'ignoreDefensive', 'ignoreOffensive', 'ignoreNegativeOffensive', 'ignorePositiveDefensive']), handlers: handlers(m)})),
	abilities: abilities.map(a => ({id: a.id, name: a.name, num: a.num, flags: a.flags, handlers: handlers(a)})),
	items: items.map(i => ({id: i.id, name: i.name, num: i.num, handlers: handlers(i), megaStone: i.megaStone, itemUser: i.itemUser})),
}));
