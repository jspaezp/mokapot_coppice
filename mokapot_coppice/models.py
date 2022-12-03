import logging
from argparse import _ArgumentGroup

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mokapot.model import Model
from mokapot.plugins import BasePlugin
from sklearn import tree
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from skranger.ensemble import RangerForestClassifier
from xgboost import XGBClassifier

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
        clf = self.get_model()
        super().__init__(clf, *args, **kwargs)

    @staticmethod
    def get_model():
        clf = LGBMClassifier(
            num_leaves=70,
            subsample=0.8,
            subsample_freq=5,
            max_depth=7,
            verbose=2,
            min_data_in_leaf=50,
            min_data_in_bin=10,
            force_row_wise=True,
        )
        return clf


class RFModel(Model):
    DESCRIPTION = "Random Forest Classifier"

    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: RF")
        clf = self.get_model()
        super().__init__(clf, *args, **kwargs)

    @staticmethod
    def get_model():
        clf = RangerForestClassifier(min_node_size=5, max_depth=15, mtry=10)
        return clf


class XGBModel(Model):
    DESCRIPTION = "XGBoost Classifier"

    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: XGBModel")
        clf = self.get_model()
        super().__init__(clf, *args, **kwargs)

    @staticmethod
    def get_model():
        return XGBClassifier()


class CatboostModel(Model):
    DESCRIPTION = "Catboost Classifier"

    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: Catboost")
        clf = self.get_model()
        super().__init__(clf, *args, **kwargs)

    @staticmethod
    def get_model():
        clf = CatBoostClassifier(
            learning_rate=0.15,
            depth=8,
            leaf_estimation_method="Newton",
            leaf_estimation_iterations=2,
            rsm=0.8,
            min_child_samples=5,
            early_stopping_rounds=10,
            metric_period=5,
            verbose=200,
            iterations=4000,
        )
        return clf


class CoppiceModel(Model):
    DESCRIPTION = (
        "Coppice Classifier, a Stacking Classifier of"
        " RF, XGB, LGBM, catboost, and Logistic Regression"
    )

    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: CoppiceModel")
        estimators = [
            ("rf", RFModel.get_model()),
            ("xgb", XGBModel.get_model()),
            ("lgbm", LGBMModel.get_model()),
            ("catboost", CatboostModel.get_model()),
            ("glm", LogisticRegression(solver="liblinear", max_iter=1000)),
        ]
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(solver="liblinear", max_iter=1000),
            passthrough=True,
            verbose=20,
        )
        super().__init__(clf, *args, **kwargs)


MODELS = {
    "ctree": CtreeModel,
    "lgbm": LGBMModel,
    "rf": RFModel,
    "xgb": XGBModel,
    "catboost": CatboostModel,
    "coppice": CoppiceModel,
}


class Plugin(BasePlugin):
    def add_arguments(parser: _ArgumentGroup) -> None:
        parser.add_argument(
            "--coppice_model",
            type=str,
            default="coppice",
            choices=MODELS.keys(),
            help=(
                "The model to use for prediction, among the models available in"
                " mokapot_coppice"
            ),
        )

    def get_model(self, config):
        if config.coppice_model is None:
            return None
        elif config.coppice_model not in MODELS:
            raise ValueError(
                f"Model {config.coppice_model} not found in mokapot_coppice"
            )
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
