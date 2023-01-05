"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""
import subprocess

import pytest
from loguru import logger

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


def test_cli_help():
    """Test that the CLI help works"""

    res = subprocess.run(["mokapot", "--help"], check=True, capture_output=True)
    assert "--coppice_model" in res.stdout.decode()


models = ["ctree", "lgbm", "rf", "xgb", "coppice", "catboost"]
grid_args = ["--coppice_with_grid", "--no-coppice_with_grid", ""]
grid_ids = ["with_grid", "no_grid", "no_grid_default"]


@pytest.mark.parametrize("grid", grid_args, ids=grid_ids)
@pytest.mark.parametrize("model", models, ids=models)
def test_cli_plugins(tmp_path, shared_datadir, model, grid):
    phospho_file = shared_datadir / "phospho_rep1.pin"

    cmd = [
        "mokapot",
        str(phospho_file),
        "-v",
        "3",
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
    ]

    cmd += ["--coppice_model", model]
    if grid:
        cmd += [grid]
    logger.info(" ".join([str(x) for x in cmd]))
    expected_msg = f"Initialising Coppice Model: {model.upper()}".upper()

    # Make sure it does not yell when the plugin is not loaded explicitly
    res = subprocess.run(cmd, check=True, capture_output=True)
    assert expected_msg not in res.stderr.decode().upper()

    assert tmp_path.joinpath("mokapot.peptides.txt").exists()
