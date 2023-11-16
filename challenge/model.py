import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from .pre_preprocess_aux import get_min_diff

from typing import Tuple, Union, List


class DelayModel:
    def __init__(self, model_filename="model.joblib"):
        self._model = LogisticRegression()  # Model should be saved in this attribute.
        self.model_file_path = os.path.join(os.path.dirname(__file__), model_filename)
        self.top_10_features = top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        self.load_model()

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        features = features[self.top_10_features]

        if target_column:
            if target_column == "delay":
                data["min_diff"] = data.apply(get_min_diff, axis=1)

                threshold_in_minutes = 15
                data["delay"] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)

            target = data[[target_column]]
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0 = len(target[target.iloc[:, 0] == 0])
        n_y1 = len(target[target.iloc[:, 0] == 1])
        class_weight = {1: n_y0 / len(target), 0: n_y1 / len(target)}

        self._model.class_weight = class_weight
        self._model.fit(features, target.values.ravel())
        joblib.dump(self._model, self.model_file_path)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        try:
            check_is_fitted(self._model)
        except NotFittedError as e:
            raise NotFittedError(
                "This DelayModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

        predicted_target = self._model.predict(features)

        return predicted_target.tolist()

    def load_model(self):
        """Loads the model from the file if it exists."""
        if os.path.exists(self.model_file_path):
            self._model = joblib.load(self.model_file_path)
