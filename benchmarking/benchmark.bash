#!/bin/bash

set -x
set -e

NUM_WORKERS=4

wandb sweep --project mokapot_coppice small_sweep_config.yml &> sweep.info
sweep_id=$(cat sweep.info | ggrep -oP "(?<=ID: ).*")

for i in {1..$NUM_WORKERS} ; do
    echo "Launching worker ${i}"
    python benchmark_agent.py --sweep_id ${sweep_id} &> log_worker_{i}.log &
done

echo "Waiting for workers to finish..."
wait
