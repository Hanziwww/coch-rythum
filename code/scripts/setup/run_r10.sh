#!/usr/bin/env bash
# Round 10 reanalyses (D113): R2 (677, joint exact test of the timer split), then R1 (333, depth-matched single-cell
# clock test). Logs in logs/; logs/r10_reanalysis.done marks the end. Run: screen -dmS r10 bash scripts/setup/run_r10.sh
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
THREADS=8 source scripts/setup/threads.sh
P="${COCH_PYTHON:-python} -u"
rm -f logs/r10_reanalysis.done
for s in 677_timer_split_joint_null 333_cell_clock_depth_matched; do
  $P scripts/$s.py > logs/$s.log 2>&1
  echo "EXIT=$?" >> logs/$s.log
done
echo R10_DONE > logs/r10_reanalysis.done
