#!/bin/bash
# replicate both arms at three initialisations; data fixed, torch init varies
set -e
arm=$1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 1 2; do
  SEED=$s python fit.py 1200 4000 scratch > /dev/null 2>&1
  MODEL=models/${arm}_s${s}_L9.pt SEED=$s python readtest.py > readtest_${arm}_s${s}.log 2>&1
  echo "$arm seed $s done"
done
