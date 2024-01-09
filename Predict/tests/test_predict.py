import unittest
import os
from Predict.predcit.run import TextPredictionModel  # Adjust this import according to your project structure

class TestTextPredictionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up any mock data or models needed for tests
        # For example, load a model or create a mock model
        model_path = "../../train/data/artefacts/2024-01-09-15-54-23"
        cls.model = TextPredictionModel.from_artefacts(model_path)
    def test_predict_valid_input(self):
        # Test prediction with valid input
        texts = ["some example text", "another example"]
        top_k = 3
        predictions = self.model.predict(texts, top_k=top_k)
        # Add assertions to check if predictions are as expected
        self.assertIsInstance(predictions, list)



if __name__ == '__main__':
    unittest.main()



from unittest.mock import MagicMock
import tempfile
import pandas as pd
from train.train import run as run_train
from preprocessing.preprocessing import utils

from Predict.predcit import run


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestPredict(unittest.TestCase):
    # TODO: CODE HERE
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            'batch_size':2,
            'epochs':1,
            'dense_dim':64,
            'min_sample_per_label':2,
            'verbose':1
            }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run_train.train(
                "fake_path",
                model_path=model_dir,
                train_conf=params,
                add_timestamp=False
                )

            model = run.TextPredictionModel.from_artefacts(model_dir)
            y_pred = model.predict(["Is it possible to execute the procedure of a function in the scope of the caller?"], 1)
            print(y_pred)

        assert  y_pred == [['php']]