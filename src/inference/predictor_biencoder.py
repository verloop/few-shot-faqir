import json
import os
import pickle
from threading import Lock

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn.preprocessing import normalize


class BiEncoderModelPredictor:
    def __init__(self, config, device: str = "cpu"):
        self.config = config
        self.model_path = self.config["INFERENCE"]["MODEL_PATH"]
        self.device = device
        self.layers_to_load = self.config["INFERENCE"]["LAYERS_TO_LOAD"]
        self.weights_lock = Lock()
        try:
            self._load(model_path=self.model_path, device=self.device)
        except Exception as e:
            print("Error in loading model")
            raise

    def _load(self, model_path, device):
        self.word_embedding_model = models.Transformer(model_path)
        self.pooling_model = models.Pooling(
            self.word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False,
        )
        self.model = SentenceTransformer(
            modules=[self.word_embedding_model, self.pooling_model], device=device
        )

        self.base_weights = {
            "pooler": self.word_embedding_model.auto_model.pooler.state_dict(),
        }

        for layer in self.layers_to_load:
            self.base_weights.update(
                {
                    f"encoder.layer.{layer}": self.word_embedding_model.auto_model.encoder.layer[
                        layer
                    ].state_dict()
                }
            )
        self.model.eval()

    def load_embeddings(self, base_npy_path: str):
        """
        Loads the npy files based on paths specified
        Args:
        base_npy_path - base path where npy files are stored
        """
        embeddings_normalized = np.load(
            base_npy_path + "/" + self.config["INFERENCE"]["EMBEDDING_FILE_NAME"]
        )
        texts = np.load(
            base_npy_path + "/" + self.config["INFERENCE"]["TEXTS_FILE_NAME"]
        )
        labels = np.load(
            base_npy_path + "/" + self.config["INFERENCE"]["LABELS_FILE_NAME"]
        )

        return embeddings_normalized, texts, labels

    def swap_classifier(self, client_weights):
        """
        Swaps the layer weights with the client weights.
        Args:
        client_weights: {"encoder.layer.{layer}}":encoder.layer.{layer},"pooler":pooler}
        """

        self.word_embedding_model.auto_model.pooler.load_state_dict(
            client_weights["pooler"]
        )

        for layer in self.layers_to_load:
            if client_weights:
                self.word_embedding_model.auto_model.encoder.layer[
                    layer
                ].load_state_dict(client_weights[f"encoder.layer.{layer}"])

                print("Swapping client weights")
            else:
                self.word_embedding_model.auto_model.encoder.layer[
                    layer
                ].load_state_dict(self.base_weights[f"encoder.layer.{layer}"])
                print("Using base weights")

    def load_client_data(self, client_id):
        client_data = {}

        layer_names_to_load = self.base_weights.keys()
        for layer in layer_names_to_load:
            weight_file_path = (
                self.config["INFERENCE"]["MODEL_DIR"]
                + f"/clients/{client_id}/classifiers/{layer}"
            )
            weights = torch.load(weight_file_path, map_location=self.device)
            client_data[layer] = weights

        print(f"Succesfully loaded client weights for {client_id}")
        return client_data

    def predict(self, client_id, top_n=3):
        base_npy_path = self.config["INFERENCE"]["MODEL_DIR"] + f"/clients/{client_id}"
        utterance = self.config["INFERENCE"]["TEXT"]
        client_weights = self.load_client_data(client_id)

        with self.weights_lock:
            try:
                self.swap_classifier(client_weights=client_weights)
            except Exception as e:
                print(f"Error in swapping client weights")
                raise

            query_embedding = self.model.encode(utterance)
            query_embedding_normalized = normalize([query_embedding], axis=1)

        try:
            embeddings_normalized, texts, labels = self.load_embeddings(
                base_npy_path=base_npy_path
            )
        except Exception as e:
            print("Error in loading embeddings")
            raise

        topn_sents, topn_similarity_scores, topn_labels = [], [], []

        if len(embeddings_normalized) == 0:
            return topn_sents, topn_similarity_scores

        try:
            sims = np.dot(query_embedding_normalized, embeddings_normalized.T)
            sims = np.squeeze(sims)
            sims_argsorted = np.argsort(sims)

            sents_sorted = list(texts[sims_argsorted])
            labels_sorted = list(labels[sims_argsorted])

            if len(sims.shape) > 0:
                sims_sorted = list(sims[sims_argsorted])
            else:
                sims_sorted = [sims]

            while len(topn_sents) < top_n and len(sents_sorted) > 0:
                sent = sents_sorted.pop()
                label = labels_sorted.pop()
                similarity = sims_sorted.pop()

                if sent not in topn_sents:
                    topn_sents.append(sent)
                    topn_labels.append(label)
                    topn_similarity_scores.append(similarity)

        except Exception as e:
            print("Error in embedding similarity")
            raise

        return topn_sents, topn_labels, topn_similarity_scores
