import argparse
import logging
from argparse import _ArgumentGroup

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mokapot.model import PERC_GRID, Model
from mokapot.plugins import BasePlugin
from sklearn import tree
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from skranger.ensemble import RangerForestClassifier
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)


class CtreeModel(Model):
    DESCRIPTION = "Decision Tree Classifier"

    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: CTree")
        clf = tree.DecisionTreeClassifier()
        super().__init__(clf, *args, **kwargs)


LGBM_ARGS = {
    "learning_rate": 0.24,
    "max_depth": 15,
    "num_leaves": 90,
    "feature_fraction": 0.65,
    "colsample_bytree": 0.65,
    "data_sample_strategy": "bagging",
    "subsample": 0.94,
    "boosting": "gbdt",
    "boosting_type": "gbdt",
    "num_iterations": 300,
    "reg_alpha": 2.5,
    "reg_lambda": 40,
    "verbose": 0,
    "min_data_in_bin": 10,
    "force_row_wise": True,
    "bagging_freq": 10,
    "subsample_freq": 10,
}


class LGBMModel(Model):
    DESCRIPTION = "LightGBM Classifier"

    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: LGBM")
        clf = self.get_model()
        super().__init__(clf, *args, **kwargs)

    @staticmethod
    def get_model():
        clf = LGBMClassifier(**LGBM_ARGS)
        return clf


class EarlyStopLGBM(LGBMClassifier):
    def fit(self, X, y, **kwargs):
        eval_subset = self.eval_subset if hasattr(self, "eval_subset") else 0.3
        print("Subsetting early stopping evaluation set")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=eval_subset)
        super().fit(X_train, y_train, eval_set=[(X_val, y_val)], **kwargs)
        return self


class EarlyStopLGBMModel(Model):
    DESCRIPTION = "LightGBM Classifier with early stopping"

    def __init__(self, *args, **kwargs):
        LOGGER.info("Initialising Coppice Model: EarlyStopLGBM")
        clf = self.get_model()
        super().__init__(clf, *args, **kwargs)

    @staticmethod
    def get_model():
        clf = EarlyStopLGBM(
            metric="cross_entropy",
            early_stopping_round=30,
            **LGBM_ARGS,
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


# very non-gently adapted from mokapot
class PercolatorGridCVModel(Model):
    def __init__(
        self,
        classifier,
        scaler=None,
        train_fdr=0.01,
        max_iter=10,
        direction=None,
        override=False,
        subset_max_train=None,
        n_jobs=-1,
    ):
        """Initialize a PercolatorModel"""
        self.n_jobs = n_jobs
        estimator = GridSearchCV(
            classifier,
            param_grid=PERC_GRID,
            refit=False,
            cv=3,
            n_jobs=n_jobs,
        )

        super().__init__(
            estimator=estimator,
            scaler=scaler,
            train_fdr=train_fdr,
            max_iter=max_iter,
            direction=direction,
            override=override,
            subset_max_train=subset_max_train,
        )


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
    "earlystop_lgbm": EarlyStopLGBMModel,
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
        parser.add_argument(
            "--coppice_with_grid",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Whether to use a grid search to find the best parameters for the"
                " coppice model (it is always used in the default svm model in mokapot)"
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

            if config.coppice_with_grid:
                LOGGER.info(f"Setting up model {config.coppice_model} with grid")
                model = PercolatorGridCVModel(
                    classifier=model_builder.get_model(),
                    train_fdr=config.train_fdr,
                    max_iter=config.max_iter,
                    direction=config.direction,
                    override=config.override,
                    subset_max_train=config.subset_max_train,
                )
            else:
                model = model_builder(
                    train_fdr=config.train_fdr,
                    max_iter=config.max_iter,
                    direction=config.direction,
                    override=config.override,
                    subset_max_train=config.subset_max_train,
                )
            return model

    def process_data(self, data, config):
        return data
