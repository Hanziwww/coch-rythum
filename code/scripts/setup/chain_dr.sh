#!/usr/bin/env bash
# Module 1 chain: metrics + feature selection + Stage B training for the remaining families.
cd "${COCH_ROOT:?set COCH_ROOT to the project root}"
P="nice -n 5 ${COCH_PYTHON:-python} -u"
$P scripts/22_dr_metrics.py cochlea > logs/22_dr_metrics_A_cochlea.log 2>&1
for fam in joint cochlea; do
  SEL=$($P scripts/20b_select_features.py $fam 2>/dev/null | tail -1)
  echo "$fam selected $SEL" >> logs/chain_dr.log
  $P scripts/21_dr_train.py $fam B $SEL > logs/21_dr_train_${fam}_B.log 2>&1
done
echo TRAIN_DONE >> logs/chain_dr.log
