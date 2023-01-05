import logging
import re
import tempfile
from pathlib import Path

import mokapot
import numpy as np
import pandas as pd
import wandb
from catboost import CatBoostClassifier
from loguru import logger
from mokapot.model import Model
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


MOD_REGEX = re.compile(r"\[.*?\]")
FLANKING_REGEX = re.compile(r"(^.{1,3}\.)|(\..{1,3}$)")


class EarlyStoppingCatboost(CatBoostClassifier):
    def fit(self, X, y, **kwargs):
        eval_subset = self.eval_subset if hasattr(self, "eval_subset") else 0.3
        print("Subsetting early stopping evaluation set")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=eval_subset)
        super().fit(X_train, y_train, eval_set=[(X_val, y_val)], **kwargs)
        return self


def run_model(data, model_args):
    clf = EarlyStoppingCatboost(
        min_child_samples=10,
        early_stopping_rounds=30,
        metric_period=1,
        leaf_estimation_method="Newton",
        verbose=200,
        **model_args,
    )

    model = Model(clf, max_iter=5)
    # Read the PSMs from the PIN file:
    psms = mokapot.read_pin(data)
    results, models = mokapot.brew(psms, model, folds=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        results.to_txt(dest_dir=tmpdir, decoys=True)
        out = get_metrics(tmpdir + "/")
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
    run = wandb.init(project="mokapot_coppice_hparams")
    artifact = run.use_artifact("jspaezp/mokapot_coppice/pinfiles:v0", type="raw_data")
    artifact_dir = artifact.download()
    DATASETS = list(Path(artifact_dir).rglob("*.pin"))

    model_args = {
        "objective": wandb.config.objective,
        "learning_rate": wandb.config.lr,
        "l2_leaf_reg": wandb.config.l2_regularization,
        "depth": wandb.config.max_depth,
        "leaf_estimation_iterations": 2,
        "subsample": wandb.config.subsample,
        "iterations": wandb.config.num_iterations,
    }

    logger.info("Starting new parameters run:")
    for k, v in model_args.items():
        logger.info(f"{k}: {v}")

    results = {}
    for data in DATASETS:
        logger.info(f"Starting new dataset {str(data)}")
        dataname = data.stem
        out = run_model(str(data), model_args=model_args)
        out.pop("table")

        # pep_dat = out.pop("table")
        # scores = [pep_dat["mokapot score"], 1-pep_dat["mokapot score"]]
        # targets = ["Target" if x else "Decoy" for x in pep_dat["Label"].array]
        # roc = wandb.plot.roc_curve(targets, np.array(scores).T)
        # wandb.log({"roc": roc})

        results[dataname] = out
        out = {dataname + k: v for k, v in out.items()}
        for k, v in out.items():
            logger.info(f"{k}: {v}")
        logger.info(out)
        wandb.log(out)

        # TODO add acceptances vs FDR plot

    mean_val = np.mean([x["npep_1pct"] for x in results.values()])
    wandb.log({"mean_npep_1pct": mean_val})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str)
    args = parser.parse_args()

    # Start sweep job.
    wandb.agent(
        args.sweep_id,
        function=main,
        project="mokapot_coppice_hparams",
        count=60,
    )
