import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "How to print the full NumPy array, without truncation?",
        "Should I use a timestamp field in mysql for timezones or other",
        "How to print the full NumPy array, without truncation?",
        "Should I use a timestamp field in mysql for timezones or other",
        "How to print the full NumPy array, without truncation?",
        "Should I use a timestamp field in mysql for timezones or other",
        "How to print the full NumPy array, without truncation?",
        "Should I use a timestamp field in mysql for timezones or other",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestTrain(unittest.TestCase):
    # TODO: CODE HERE
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_train(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            "batch_size": 2,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 2,
            "verbose": 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run.train(
                "fake_path",
                model_path=model_dir,
                train_conf=params,
                add_timestamp=True
                )

        # TODO: CODE HERE
        # assert that accuracy is equal to 1.0
        assert accuracy == 1

