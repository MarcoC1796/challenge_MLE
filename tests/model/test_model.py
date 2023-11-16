import unittest
import os
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel


class TestModel(unittest.TestCase):
    FEATURES_COLS = [
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

    TARGET_COL = ["delay"]

    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        data_file_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "data.csv"
        )
        data_types = {
            "Fecha-I": str,  # Scheduled date and time of the flight.
            "Vlo-I": str,  # Scheduled flight number.
            "Ori-I": str,  # Programmed origin city code.
            "Des-I": str,  # Programmed destination city code.
            "Emp-I": str,  # Scheduled flight airline code.
            "Fecha-O": str,  # Date and time of flight operation.
            "Vlo-O": str,  # Flight operation number of the flight.
            "Ori-O": str,  # Operation origin city code.
            "Des-O": str,  # Operation destination city code.
            "Emp-O": str,  # Airline code of the operated flight.
            "DIA": int,  # Day of the month of flight operation.
            "MES": int,  # Number of the month of operation of the flight.
            "AÑO": int,  # Year of flight operation.
            "DIANOM": str,  # Day of the week of flight operation.
            "TIPOVUELO": str,  # Type of flight, I = International, N = National.
            "OPERA": str,  # Name of the airline that operates.
            "SIGLAORI": str,  # Name city of origin.
            "SIGLADES": str,  # Destination city name.
        }
        self.data = pd.read_csv(filepath_or_buffer=data_file_path, dtype=data_types)

    def test_model_preprocess_for_training(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)

    def test_model_preprocess_for_serving(self):
        features = self.model.preprocess(data=self.data)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_model_fit(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")

        _, features_validation, _, target_validation = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        self.model.fit(features=features, target=target)

        predicted_target = self.model._model.predict(features_validation)

        report = classification_report(
            target_validation, predicted_target, output_dict=True
        )

        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30

    def test_model_predict(self):
        features = self.model.preprocess(data=self.data)

        predicted_targets = self.model.predict(features)

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(
            isinstance(predicted_target, int) for predicted_target in predicted_targets
        )
