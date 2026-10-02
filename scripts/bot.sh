#!/usr/bin/env zsh
# The Showdown bot (run on the laptop, not ilab). N counts matches: Bo3 series, or Bo1 games.
#
#   scripts/bot.sh --ladder [N]            ladder N matches; no N: until Ctrl-C (finishes the current
#                                          game or Bo3 series first; Ctrl-C twice quits at once)
#   scripts/bot.sh --challenge NAME [N]    challenge NAME to N matches (default 1); they accept
#   scripts/bot.sh --accept [NAME...]      take challenges until Ctrl-C, from anyone or these names
#
#   --bo3 (default) / --bo1                match type; --accept takes both either way
#   --cts                                  Bo1 with closed team sheets; search tries 4 guesses at their sets
#   --mb                                   Reg M-B instead of Reg M-C
#   --general                              M-C: the top-10 pool with mc-bo1-rnad-v3 instead of MC196 with mc-bo1-rnad-v3-mc196
#   --top N                                M-C: like --general, but a random one of the best N teams per match
#   --use TEAM...                          M-C: a random one of these reg_mc_top teams per match (e.g. --use MC408 MC358)
#   --gauntlet [N]                         M-C: the top N (default 10) teams each play 4 series, the worse
#                                          half is dropped, repeat until 2 are left (then those two).
#                                          Carries on across restarts; --gauntlet-new starts it over
#   --watch                                a local page that shows the live battle and moves on to
#                                          game 2/3 and the next match by itself (links always print)
#   --no-timer                             don't turn the battle timer on (on by default)
#   --no-munchstats                        don't refresh MunchStats usage (search's set and spread guesses)
#
# Laddering and taking challenges at once: a second terminal with a second account, e.g.
#   BOT=nimnimbot2 scripts/bot.sh --accept --top 3
# (one account can't do both: each process would try to play the other's battles).
# Environment: BOT (default nimnimbot), PS_PASSWORD (asked if unset; blank if unregistered),
# SERVER=local for a local test server. Unknown --flags go to the player (e.g. --port 8000).
# Games: data/live_games/<date>/ (replays + games.jsonl) and data/bot_logs/ (training layout).
set -e
self=${0:A}
cd "${self:h}/.."

usage() { sed -n '2,15p' $self | sed 's/^# \{0,1\}//'; exit 1; }

mode= name= n= bo=bo3 reg=regmc watch=0 general=0 top= gauntlet=0 names=() use=() pass=()
while (( $# )); do
  case $1 in
    --ladder)    mode=ladder ;;
    --challenge) mode=challenge; name=$2; shift ;;
    --accept)    mode=accept ;;
    --bo3)       bo=bo3 ;;
    --bo1)       bo= ;;
    --cts)       bo=; pass+=(--closed-sheets --search-worlds 4) ;;
    --mb)        reg=regmb ;;
    --watch)     watch=1 ;;
    --general)   general=1 ;;
    --top)       general=1; top=$2; shift ;;
    --use)       general=1; while [[ $# -gt 1 && $2 != --* ]]; do use+=($2); shift; done ;;
    --gauntlet)  general=1; gauntlet=1; top=10; [[ $# -gt 1 && $2 == <-> ]] && { top=$2; shift } ;;
    --gauntlet-new) rm -f data/live_games/gauntlet.since ;;
    -h|--help)   usage ;;
    --*)         pass+=($1); [[ $# -gt 1 && $2 != --* ]] && { pass+=($2); shift } ;;
    *)           if [[ $mode == accept ]]; then names+=($1); else n=$1; fi ;;
  esac
  shift
done
[[ -n $mode ]] || usage
[[ $mode != challenge || -n $name ]] || { echo "--challenge needs a NAME"; exit 1; }

format=gen9championsvgc2026$reg$bo
# reg_mc_top best first (team_rank round 2, docs/training-ideas.md); the last three weren't ranked.
ranked=(MC196 MC147 MC371 MC378 MC358 MC408 MC337 MC321 MC41 MC4)
if [[ $reg == regmc ]] && (( general )); then
  data=(--showdown pokemon-showdown-mc --teams data/teams/reg_mc_top --sets data/teams/reg_mc --model data/models/mc-bo1-rnad-v3.pt)
  [[ -n $top ]] && data+=(--only ${ranked[1,top]})
  (( ${#use} )) && data+=(--only $use)
  if (( gauntlet )); then
    since=data/live_games/gauntlet.since  # when this gauntlet started: games before it don't count
    [[ -f $since ]] || date '+%Y-%m-%d %H:%M:%S' >$since
    data+=(--gauntlet 4 2 --gauntlet-since "$(<$since)")
    echo "gauntlet since $(<$since): ${ranked[1,top]}"
  fi
elif [[ $reg == regmc ]]; then
  data=(--showdown pokemon-showdown-mc --teams data/teams/mc196 --sets data/teams/reg_mc --model data/models/mc-bo1-rnad-v3-mc196.pt)
else
  data=(--model data/models/archive/mb-bo3-rnad-v3.pt)
fi
common=(--accept ${BOT:-nimnimbot} --server ${SERVER:-showdown} --format $format $data
        --search --opp-model data/models/all-bo3-opp-v2.pt --search-prior 0.1 --greedy --vary --vary-k 1)
(( watch )) && common+=(--watch)
forever=1000000  # "until Ctrl-C"

# Refresh MunchStats usage (spreads, natures, sets) older than a week in the background, most used
# species first; search re-reads it as it lands. Its log: data/live_games/munchstats.log.
if [[ ${SERVER:-showdown} == showdown ]] && (( ! ${pass[(I)--no-munchstats]} )); then
  mkdir -p data/live_games
  uv run python -m ai_vgc.munchstats >>data/live_games/munchstats.log 2>&1 &!
fi

# Ask for the password once, so a reconnect (below) doesn't ask again.
if [[ ${SERVER:-showdown} == showdown && -z ${PS_PASSWORD+x} ]]; then
  read -rs "PS_PASSWORD?Showdown password for ${BOT:-nimnimbot} (blank if unregistered): "; echo
  export PS_PASSWORD
fi

# Exit code 75: the connection dropped (player.py's watchdog); log in again and carry on.
run() {
  local code
  while true; do
    code=0
    uv run --extra nn python -m ai_vgc.nn.player $common "$@" $pass || code=$?  # (set -e)
    (( code == 75 )) || return $code
    echo "reconnecting in 10 s (Ctrl-C to stop)"
    sleep 10
  done
}
case $mode in
  ladder)    run --ladder --n ${n:-$forever} ;;
  challenge) run --challenge $name --n ${n:-1} ;;
  accept)    if (( ${#names} )); then run --n $forever --from $names; else run --n $forever; fi ;;
esac
