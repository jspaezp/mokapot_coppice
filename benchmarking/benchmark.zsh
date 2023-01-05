#!/usr/bin/zsh

set -x
set -e

# Copied shamelessly from stackoverflow
command_exists () {
    type "$1" &> /dev/null ;
}

NUM_WORKERS=1

wandb sweep --project mokapot_coppice sweep_config.yml &> sweep.info

# Added here to support OSX where gnu grep is installed as ggrep
if command_exists ggrep ; then
    sweep_id=$(cat sweep.info | ggrep -oP "(?<=ID: ).*")
else
    sweep_id=$(cat sweep.info | grep -oP "(?<=ID: ).*")
fi


for i in {1..$NUM_WORKERS} ; do
    echo "Launching worker ${i}"
    python benchmark_agent.py --sweep_id ${sweep_id} &> log_worker_${i}.log &
done

echo "Waiting for workers to finish..."
wait
