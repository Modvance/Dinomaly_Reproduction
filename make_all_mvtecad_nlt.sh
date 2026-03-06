#!/usr/bin/env bash

for seed in {01..03}; do
  python make_mvtecad_nlt.py \
    --source-dir ../mvtec_anomaly_detection \
    --dest-dir   ../LTN_datasets/mvtecad-step_k1-seed${seed} \
    --prune-manifest manifest/mvtecad-nlt/step_k1/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/mvtecad-nlt/step_k1/seed${seed}/inject_defects.txt
done

for seed in {01..03}; do
  python make_mvtecad_nlt.py \
    --source-dir ../mvtec_anomaly_detection \
    --dest-dir   ../LTN_datasets/mvtecad-step_k4-seed${seed} \
    --prune-manifest manifest/mvtecad-nlt/step_k4/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/mvtecad-nlt/step_k4/seed${seed}/inject_defects.txt
done

for seed in {01..03}; do
  python make_mvtecad_nlt.py \
    --source-dir ../mvtec_anomaly_detection \
    --dest-dir   ../LTN_datasets/mvtecad-pareto-seed${seed} \
    --prune-manifest manifest/mvtecad-nlt/pareto/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/mvtecad-nlt/pareto/seed${seed}/inject_defects.txt
done