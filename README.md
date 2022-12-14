
![](assets/coppice2.jpg)
- Dissclaimer: image generated using stable diffusion

# Mokapot Coppice

This project is a mokapot plugin that adds tree-based models to the mokapot framework.

## Installation

Right now the package is not in pypi so we will have to install it manually from github.

### Pipx installation

```
pipx install mokapot
pipx inject mokapot git+https://github.com/jspaezp/mokapot_coppice.git
```

### Venv installation

```shell
python -m venv venv
source venv/bin/activate

python -m pip install mokapot git+https://github.com/jspaezp/mokapot_coppice.git
```

## Usage

Once everything is installed, we just pass the `--coppice_model {model}` option
to the cli!

```shell
# the model can be any of "ctree", "lgbm", "rf", "xgb", "catboost"
mokapot {myfile.pin} --plugin mokapot_coppice --coppice_model xgb
```

# Issues

- Right now coverage is not meassured correctly due to tests calling an external process.
