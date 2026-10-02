# Team ranking (Reg M-C, CTS Bo1)

As of 2026-10-02. Every team below has its own v7 specialist (`data/models/mc-cts-rnad-v7-<team>.pt`).

- **Round robin:** the 19 teams against each other, each piloted by its own v7 (job 414981, 100 series per pair, 1,800 each, about ±2.3 points).
- **Fair test:** the team's v7 against all 378 pool teams, with all-bo3-bc-v2 piloting the pool (`team_test --closed-sheets`, 3,024 games, about ±1 point). Closest thing we have to ladder. "–" means not tested yet.
- **Style** is a reading of the six Pokémon, not something the data records.

Ladder plan: one rain team only. MC378 is the rain pick (near the top of both tests). MC408 and MC358 are the same rain core and are dropped. The second team comes from the non-rain candidates in bold, once they've had the fair test.

| # | Team | Style | Pokémon | Round robin | Fair test |
|---|---|---|---|---|---|
| 1 | U1 | Trick Room | Incineroar / Maushold / Sinistcha / Sneasler / Blastoise / Delphox | 65.8 | 87.0 |
| 2 | **MC378** | **Rain (ladder pick)** | Archaludon / Pelipper / Grimmsnarl / Charizard / Golisopod / Dragapult | 64.6 | **96.4** |
| 3 | MC408 | Rain | Politoed / Golisopod / Archaludon / Farigiraf / Charizard / Grimmsnarl | 57.8 | 95.3 |
| 4 | **MC301** | Sun + Trick Room | Rillaboom / Sylveon / Charizard / Incineroar / Farigiraf / Garchomp | 57.6 | – |
| 5 | MC371 | Rain | Golisopod / Basculegion / Pelipper / Archaludon / Sneasler / Salamence | 56.3 | – |
| 6 | MC196 | Rain | Pelipper / Incineroar / Rillaboom / Archaludon / Annihilape / Golisopod | 53.2 | – |
| 7 | **U6** | Sun + Trick Room | Charizard / Kingambit / Sylveon / Farigiraf / Garchomp / Aerodactyl | 52.8 | – |
| 8 | MC147 | Rain | Swampert / Golisopod / Pelipper / Archaludon / Farigiraf / Sneasler | 52.7 | – |
| 9 | U4 | Rain | Golisopod / Pelipper / Archaludon / Grimmsnarl / Garchomp / Basculegion | 52.6 | – |
| 10 | **MC41** | Tailwind offence | Salamence / Sylveon / Kingambit / Basculegion / Sneasler / Rillaboom | 52.3 | – |
| 11 | MC358 | Rain | Archaludon / Golisopod / Farigiraf / Politoed / Charizard / Grimmsnarl | 50.0 | 96.0 |
| 12 | **MC4** | Sand / bulky | Golisopod / Incineroar / Tyranitar / Sneasler / Sinistcha / Milotic | 50.0 | – |
| 13 | P1 | Sun + Trick Room | Salamence / Charizard / Annihilape / Farigiraf / Mimikyu / Kingambit | 49.3 | – |
| 14 | MC321 | Offence | Salamence / Raichu / Gholdengo / Incineroar / Sneasler / Rillaboom | 43.4 | – |
| 15 | MC388 | Aurora Veil offence | Rillaboom / Kingambit / Froslass / Raichu / Arcanine-H / Sneasler | 40.7 | 91.3 |
| 16 | MC272 | Tailwind offence | Raichu / Rillaboom / Arcanine-H / Sylveon / Gholdengo / Staraptor | 39.5 | 92.4 |
| 17 | MC222 | Tailwind offence | Salamence / Floette / Rillaboom / Sneasler / Kingambit / Basculegion | 39.2 | – |
| 18 | MC354 | Tailwind offence | Floette / Incineroar / Gholdengo / Sneasler / Rillaboom / Dragonite | 37.4 | – |
| 19 | MC337 | Offence | Raichu / Salamence / Primarina / Rillaboom / Arcanine-H / Gholdengo | 34.8 | – |

## Ladder (CTS Bo1, each team with its own v7)

| Team | Record | Peak | Latest (as of 2026-10-02) |
|---|---|---|---|
| MC301 | 72–59 | 1646 | 1614 |
| MC378 | 30–24 | 1568 | 1484 |

## Next

Fair-test the non-rain candidates (ilab):

```
cd ~/ai-vgc && for t in MC301 U6 MC41 MC4; do env POOL=data/teams/reg_mc SHARDS=4 sbatch --exclude=ilab1 scripts/slurm/nn_team_test.sh data/teams/reg_mc/$t.txt --model data/models/mc-cts-rnad-v7-$t.pt --opponent-model data/models/all-bo3-bc-v2.pt --format gen9championsvgc2026regmc --showdown pokemon-showdown-mc --closed-sheets; done
```

Results:

```
cd ~/ai-vgc && for f in $(ls logs/vgc-nn-team-test-*.out | sort -V | tail -4); do echo "$(grep -m1 'reg_mc/' $f | cut -c1-60) $(grep -o 'overall.*' $f)"; done
```

The one closest to MC378's 96.4% becomes the second ladder team.
