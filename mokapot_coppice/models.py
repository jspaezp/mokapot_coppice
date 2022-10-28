import logging
from argparse import _ArgumentGroup

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.enemble import StackingClassifier
from skranger.ensemble import RangerForestClassifier
from xgbosst import XGBClassifier
from lightgbm import LGBMClassifier

from mokapot.model import Model
from mokapot.plugins import BasePlugin

LOGGER = logging.getLogger(__name__)


class CtreeModel(Model):
    DESCRIPTION = "Decision Tree Classifier"
    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: CTree")
        clf = tree.DecisionTreeClassifier()
        super().__init__(clf, *args, **kwargs)

class LGBMModel(Model):
    DESCRIPTION = "LightGBM Classifier"
    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: LGBM")
        clf = LGBMClassifier()
        super().__init__(clf, *args, **kwargs)

class RFModel(Model):
    DESCRIPTION = "Random Forest Classifier"
    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: RF")
        clf = RangerForestClassifier()
        super().__init__(clf, *args, **kwargs)

class XGBModel(Model):
    DESCRIPTION = "XGBoost Classifier"
    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: XGBModel")
        clf = XGBClassifier()
        super().__init__(clf, *args, **kwargs)

class CoppiceModel(Model):
    DESCRIPTION = (
        "Coppice Classifier, a Stacking Classifier of"
        " RF, XGB, LGBM, and Logistic Regression")
    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: CoppiceModel")
        estimators = [
            ('rf', RFModel()),
            ('xgb', XGBModel()),
            ('lgbm', LGBMModel()),
            ('glm', LogisticRegression()),
        ]
        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        super().__init__(clf, *args, **kwargs)


MODELS = {
    "ctree": CtreeModel,
    "lgbm": LGBMModel,
    "rf": RFModel,
    "xgb": XGBModel,
    "coppice": CoppiceModel,
}


class Plugin(BasePlugin):
    def add_arguments(parser: _ArgumentGroup) -> None:
        parser.add_argument(
            "--coppice_model",
            type=str,
            default="coppice",
            choices=MODELS.keys(),
            help="The model to use for prediction, among the models available in mokapot_coppice",
        )

    def get_model(self, config):
        if config.coppice_model is None:
            return None
        elif config.coppice_model not in MODELS:
            raise ValueError(f"Model {config.coppice_model} not found in mokapot_coppice")
        else:
            LOGGER.info(f"Using model {config.coppice_model}")
            model_builder = MODELS[config.coppice_model]
            return model_builder(
                train_fdr=config.train_fdr,
                max_iter=config.max_iter,
                direction=config.direction,
                override=config.override,
                subset_max_train=config.subset_max_train,
            )

    def process_data(self, data, config):
        return data
