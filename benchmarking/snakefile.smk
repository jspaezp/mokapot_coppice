
from snakemake.utils import min_version
min_version("6.0")

from contextlib import contextmanager
from loguru import logger as lg_logger

import tempfile
import requests
import re
from pprint import pformat as pfm

import numpy as np
import pandas as pd

lg_logger.info(f"Config: {pfm(config)}")

def split_configs(config: dict):
    """Split config into multiple configs.

    The config is split into 3 main configurations:
    1. The default configuration, which is applied to all the files if it is not
         overridden by the other configurations.
    2. The 'experiment' configuration, which is applied to all the files in the
         experiment.
         - The 'experiment' bundles files that share search parameters and database.

    Returns:
        tuple[dict, dict]: The default, experiment configurations.
    """
    DEFAULT_CONFIG = config["default"]
    experiment_configs = {x["name"]: DEFAULT_CONFIG.copy() for x in config["experiments"]}
    for x in config["experiments"]:
        experiment_configs[x["name"]].update(x)

    lg_logger.info(f"Experiment Configs:\n {pfm(experiment_configs)}")
    return DEFAULT_CONFIG, experiment_configs


def expand_files(experiment_configs):
    """Expand the files in the experiment configurations.

    Args:
        experiment_configs (dict): The experiment configurations.

    Returns:
        tuple[dict, dict]: The expanded experiment configurations.
    """

    files = []
    exp_files_map = {x: {y:[] for y in ('mzML', 'pin')} for x in experiment_configs}

    for experiment, exp_values in experiment_configs.items():
        fasta_name = Path(exp_values['fasta']['value']).stem
        files.append(f"results/{experiment}/fasta/{fasta_name}.fasta")
        tmp_psm_files = exp_values["files"]
        tmp_psm_files = [Path(x).stem for x in tmp_psm_files]
        exp_files_map[experiment]['mzML'] = exp_values["files"]
        for raw_file in tmp_psm_files:
            pin_file = f"results/{experiment}/comet/{raw_file}.pin"
            exp_files_map[experiment]['pin'].append(pin_file)
            files.append(pin_file)


    lg_logger.info(f"Expanded files: {files}")
    for f in files:
        assert " " not in f, f"No spaces are allowed in file names. {f} has a space."
    return files, exp_files_map


DEFAULT_CONFIG, experiment_configs = split_configs(config)
files, exp_files_map = expand_files(experiment_configs)
PINFILES = [x for x in files if x.endswith("pin")]


rule all:
    input:
        *files


from pathlib import Path

for exp_name in experiment_configs:
    for filetype in ['raw','mzml','comet','fasta','mokapot','bibliospec']:
        Path(f"results/{exp_name}/{filetype}").mkdir(parents=True, exist_ok=True)

# rule convert_raw:
#     """
#     Convert raw files to mzml
#
#     It is currently not implemented.
#     I am using docker to run msconvert in GCP
#     """
#     input:
#         "results/{experiment}/raw/{raw_file}",
#     output:
#         "results/{experiment}/mzml/{raw_file}.mzML"
#     run:
#         raise NotImplementedError
# ruleorder: link_mzML > convert_raw

def get_provided_file(wildcards):
    """Gets the location of the raw file from the
    configuration.
    """
    provided_files = experiment_configs[wildcards.experiment]['files']
    out = [x for x in provided_files if Path(x).stem == wildcards.raw_file]
    assert len(out) == 1, f"Could not find {wildcards.raw_file} in {provided_files}"
    if not out[0].endswith('.mzML'):
        raise NotImplementedError("Only mzML files are supported.")
    if not Path(out[0]).exists():
        lg_logger.warning(f"Provided file {out[0]} does not exist.")

    return out[0]

rule link_mzML:
    """
    Link mzML

    links an mzML from anywhere in the local computer to the mzml sub-directory
    """
    input:
        in_mzml = get_provided_file
    output:
        out_mzml = "results/{experiment}/mzml/{raw_file}.mzML"
    run:
        # The actual path for the link
        lg_logger.info("Linking mzML file")
        link = Path(output.out_mzml)

        lg_logger.info("Creaiting dir"+ f"mkdir -p {str(link.parent.resolve())}")
        # cmd = f"mkdir -p {str(link.parent.resolve())}"
        # shell(cmd)

        shell_cmd = f"ln -v -s  {str(Path(input.in_mzml).resolve())} {str(link.resolve())}"
        lg_logger.info(f"Shell Link ${shell_cmd}")
        shell(shell_cmd)
        lg_logger.info(f"Created link {link} to {input.in_mzml}")


rule get_fasta:
    """Gets fasta files needed for experiments

    Download the fasta file from the internet if its an internet location.
    It it exists locally it just copies it to the results folder
    """
    output:
        fasta_file = "results/{experiment}/fasta/{fasta_file}.fasta",
    run:
        fasta_conf = dict(experiment_configs[wildcards.experiment]['fasta'])
        lg_logger.info(f"Getting fasta file: {fasta_conf}")

        fasta_name = fasta_conf['value']
        fasta_type = fasta_conf['type']
        lg_logger.info(f"Getting fasta file: {fasta_conf['value']}")
        lg_logger.info(f"Getting fasta file: {fasta_conf['type']}")

        if fasta_type.startswith("url"):
            lg_logger.info("Fasta of type url")
            shell(f"wget {fasta_name} -O {output.fasta_file}")
        elif fasta_type.startswith("file"):
            lg_logger.info("Fasta of type file")
            shell(f"cp {fasta_name} {output.fasta_file}")
        elif fasta_type.startswith("uniprot"):
            lg_logger.info("Fasta of type uniprot")
            url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28proteome%3A{PROTEOME}%29%20AND%20%28reviewed%3Atrue%29"
            url = url.format(PROTEOME=fasta_name)
            lg_logger.info(url)
            lg_logger.info(f"Cache not found for {fasta_name}. Downloading from uniprot {url}.")
            all_fastas = requests.get(url).text

            with open(output.fasta_file, "w") as f:
                f.write(all_fastas)

        else:
            lg_logger.info(f"{fasta_type} not recognized as a fasta type")
            raise Exception(f"Unknown fasta type {fasta_type}")


def update_comet_config(infile, outfile, config_dict: dict):
    """
    Updates the values in a comet config file

    Reads the configuration in the infile and
    writes the updated configuration to the outfile
    """
    lg_logger.info(f"Updating comet file {infile} to {outfile}, with {config_dict}")
    with open(infile, "r") as f:
        with open(outfile, "w+", encoding="utf-8") as of:
            for line in f:
                for key in config_dict:
                    if line.startswith(key):
                        lg_logger.debug(str(line))
                        line = f"{key} = {config_dict[key]}\n"
                        lg_logger.debug(str(line))

                of.write(line)
    lg_logger.info(f"Done updating comet file {infile} to {outfile}, with {config_dict}")
    return outfile


rule generate_comet_config:
    """Generates comet config for each experiment

    Generates a comet config file by using the default config
    generated by `comet -p` and updating it with the values
    in the config.yml file
    """
    output:
        param_file = "results/{experiment}/{experiment}.comet.params",
    shadow: "minimal"
    run:
        lg_logger.info("Getting default comet params")
        shell(f"set -x; set -e ; set +o pipefail; out=$(comet -p) ; rc=$? ; echo 'Exit code for comet was ' $rc ; exit 0 ")
        lg_logger.info("Done running shell")
        comet_config = experiment_configs[wildcards.experiment]["comet"]
        lg_logger.info(f"Updating comet file comet.params.new to {output}, with {comet_config}")
        update_comet_config("comet.params.new", output.param_file, comet_config)
        lg_logger.info("Done updating expmt shell")


def get_fasta_name(wildcards):
    """
    Gets the correct fasta name and path
    from the experiment config
    """
    fasta_name = Path(experiment_configs[wildcards.experiment]['fasta']['value']).stem
    out = f"results/{str(wildcards.experiment)}"
    out += f"/fasta/{str(fasta_name)}.fasta"
    return out


rule comet_search:
    """Uses comet to search a single mzml file

    Every run takes ~1.5 CPU hours, so in a 20 CPU machine, every file takes 5 minutes.
    Usually 16gb of memory is more than enough for any file.
    """
    input:
        fasta=get_fasta_name,
        mzml="results/{experiment}/mzml/{raw_file}.mzML",
        comet_config = "results/{experiment}/{experiment}.comet.params",
    output:
        pin="results/{experiment}/comet/{raw_file}.pin",
    params:
        base_name="results/{experiment}/comet/{raw_file}",
    threads: 20
    run:
        lg_logger.info("Starting Comet")
        update_dict = {
            "num_threads": threads,
            "output_sqtfile": 0,
            "output_txtfile": 0,
            "output_pepxmlfile": 0,
            "output_mzidentmlfile": 0,
            "output_percolatorfile": 1,
            "max_variable_mods_in_peptide":3,
            "num_output_lines": 5,
            "clip_nterm_methionine": 0,
            "decoy_search": 2 }
        handle, tmp_config = tempfile.mkstemp("config", dir = f"results/{wildcards.experiment}/comet/")

        lg_logger.info(f"Updating Params with: {update_dict}")
        update_comet_config(str(input.comet_config), str(tmp_config), update_dict)

        shell_cmd = f"comet -P{tmp_config} -D{input.fasta} -N{params.base_name} {input.mzml}"
        lg_logger.info(f"Executing {shell_cmd}")
        shell(shell_cmd)


rule upload_wandb_artifact:
    input:
      *PINFILES,
    output: touch("uploaded.done")
    run:
      run = wandb.init(project="my_project")
      my_data = wandb.Artifact("new_dataset", type="raw_data")
      my_data.add_dir("path/to/my/data")
      run.log_artifact(my_data)
