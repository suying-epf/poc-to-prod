import json
import argparse
import os
import time
import numpy as np
from collections import OrderedDict

from keras.models import load_model
from numpy import argsort

from preprocessing.preprocessing.embeddings import embed

import logging

logger = logging.getLogger(__name__)


class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.labels_to_index = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.labels_to_index.items()}

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # TODO: CODE HERE
        model_path = os.path.join(artefacts_path, "model.h5")
        params_path = os.path.join(artefacts_path, "params.json")
        labels_index_path = os.path.join(artefacts_path, "labels_index.json")
        # load model
        model = load_model(model_path)

        # TODO: CODE HERE
        # load params
        with open(params_path, 'r') as file:
            params = json.load(file)

        # TODO: CODE HERE
        # load labels_to_index
        with open(labels_index_path, 'r') as file:
            labels_to_index = json.load(file)

        labels_index_inv = {v: k for k, v in labels_to_index.items()}

        return cls(model, params, labels_index_inv)

    def predict(self, text_list, top_k=5):
        """
            predict top_k tags for a list of texts
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict
        """
        tic = time.time()
        print("Label Index Mapping:", self.labels_index_inv)

        logger.info(f"Predicting text_list=`{text_list}`")

        # TODO: CODE HERE
        # embed text_list
        embeddings = embed(text_list)

        # TODO: CODE HERE
        # predict tags indexes from embeddings
        predictions_raw = self.model.predict(embeddings)

        # TODO: CODE HERE
        # from tags indexes compute top_k tags for each text
        top_k_indices = np.argsort(predictions_raw, axis=1)[:, -top_k:]
        print("Top K Indices:", top_k_indices)

        try:
            predictions = [[self.labels_index_inv[str(idx)] for idx in single_prediction] for single_prediction in top_k_indices]

        except KeyError as e:
            print("KeyError with index:", e)
            print("Available keys in labels_index_inv:", self.labels_index_inv.keys())
            raise


        logger.info("Prediction done in {:2f}s".format(time.time() - tic))

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))
