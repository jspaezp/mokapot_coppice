import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

MOD_REGEX = re.compile(r"\[.*?\]")
FLANKING_REGEX = re.compile(r"(^.{1,3}\.)|(\..{1,3}$)")


def run_model(model, data, grid=False):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)
        run = f"python -m mokapot.mokapot {data} --keep_decoys --dest_dir {tmpdirname}"
        if model != "baseline":
            run += f" --plugin mokapot_coppice --coppice_model {model}"
            if grid:
                run += " --coppice_with_grid"

        print(run)
        subprocess.run(run, shell=True, check=True)
        out = get_metrics(tmpdirname+ "/")
    return out


def get_metrics(outdir):
    pep_dat = pd.concat(
        [
            pd.read_csv(outdir + "mokapot.decoy.peptides.txt", sep="\t"),
            pd.read_csv(outdir + "mokapot.peptides.txt", sep="\t"),
        ]
    )
    psm_dat = pd.concat(
        [
            pd.read_csv(outdir + "mokapot.decoy.psms.txt", sep="\t"),
            pd.read_csv(outdir + "mokapot.psms.txt", sep="\t"),
        ]
    )

    one_percent_peps = pep_dat["Peptide"][
        pep_dat["Label"] * (pep_dat["mokapot q-value"] < 0.01)
    ]
    npep = len(one_percent_peps)
    npsm = len(
        psm_dat["Peptide"][psm_dat["Label"] * (psm_dat["mokapot q-value"] < 0.01)]
    )
    sequences = [MOD_REGEX.sub("", x) for x in one_percent_peps]
    sequences = [FLANKING_REGEX.sub("", x) for x in sequences]
    nseq = len(set(sequences))

    return {"npep_1pct": npep, "npsm_1pct": npsm, "nseq_1pct": nseq, "table": pep_dat}


def main():
    run = wandb.init(project="mokapot_coppice")
    artifact = run.use_artifact("jspaezp/mokapot_coppice/pinfiles:v0", type="raw_data")
    artifact_dir = artifact.download()
    DATASETS = list(Path(artifact_dir).rglob("*.pin"))

    model = wandb.config.model
    grid = wandb.config.grid

    results = {}
    for data in DATASETS:
        dataname = data.stem
        out = run_model(model, str(data), grid=grid)
        out.pop("table")

        # pep_dat = out.pop("table")
        # scores = [pep_dat["mokapot score"], 1-pep_dat["mokapot score"]]
        # targets = ["Target" if x else "Decoy" for x in pep_dat["Label"].array]
        # roc = wandb.plot.roc_curve(targets, np.array(scores).T)
        # wandb.log({"roc": roc})

        results[dataname] = out
        out = {dataname + k: v for k, v in out.items()}
        wandb.log(out)

        # TODO add acceptances vs FDR plot

    mean_val = np.mean([x["npep_1pct"] for x in results.values()])
    wandb.log({"mean_npep_1pct": mean_val})


# Define sweep config
# sweep_configuration = {
#     'method': 'grid',
#     'name': 'sweep',
#     'metric': {'goal': 'maximize', 'name': 'mean_npep_1pct'},
#     'parameters':
#     {
#         'model': {'values': ["baseline", "ctree", "lgbm", "rf", "xgb", "catboost"]},
#      }
# }
#
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="mokapot_coppice")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str)
    args = parser.parse_args()

    # Start sweep job.
    wandb.agent(args.sweep_id, function=main, project="mokapot_coppice")
