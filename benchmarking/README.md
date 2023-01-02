
# Benchmarking Data

The idea of this repo is to have a reproducible way of generating .pin file to use
for the benchmarking of the models/parameters in the repository.

# Run assumptions

- comet has been downloaded and in your path.
- mzML file have been downloaded from the GCP repository.
- When uploading data to wandb, have the `WANDB_API_KEY=$YOUR_API_KEY` variable set

# Running

```
snakemake --verbose --cores 4 --directory $PWD -s snakefile.smk --keep-incomplete --configfile config.yml
```

The outcome of this run will be a series of .pin files, which also will be uploaded
to wandb (since git does not want/handle well large data).

These files will then be used in CICD to test the performance of the models.

# Adding files to the testing

If you want to add more files to the workflow please reach out! I would love to have
a better representation of the data available form the proteomics community!
Feel free to open and issue and leave a brief explanation on what files you want added
and (maybe...) why you feel like that kind of data has not been represented.
