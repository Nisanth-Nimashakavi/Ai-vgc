#!/bin/bash
# Progress of the last scripts/slurm/teams_pipeline.sh run:
#   cd ~/ai-vgc && bash scripts/slurm/teams_status.sh
cd ~/ai-vgc
# "stage1=A stage2=B rr=C baseline=D" (teams_pipeline.sh) or "stage1 A stage2 B rr C baseline D"
line=$(grep '^stage1' logs/team_jobs.txt | tail -1 | sed -E 's/(stage1|stage2|rr|baseline) /\1=/g')
[[ -n $line ]] || { echo "no pipeline in logs/team_jobs.txt"; exit 1; }
eval "$line"
echo "$line"
echo
sacct -j "$stage1,$stage2,$rr,$baseline" -X --format=JobID%16,JobName%18,State,Elapsed | grep -v -- '----'
echo
echo "== RL progress (last iteration per team)"
for f in logs/vgc-nn-team-rl-"$stage1"_*.out logs/vgc-nn-team-rl-"$stage2"_*.out; do
  [[ -f $f ]] || continue
  printf '%-40s %s\n' "$(grep -m1 '^stage' "$f")" "$(grep '^iter' "$f" | tail -1 | cut -c1-90)"
done
for j in $baseline $rr; do
  f=logs/vgc-nn-team-rr-$j.out
  [[ -s $f ]] || continue
  echo
  echo "== round robin $j ($([[ $j == "$baseline" ]] && echo 'baseline, one model' || echo 'specialists'))"
  cat "$f"
done
