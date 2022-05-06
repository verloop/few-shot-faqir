""" Embedding approaches """

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseEmbeddings:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[
            input_mask_expanded == 0
        ] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]

    def cls_pooling(self, model_output):
        return model_output[0][:, 0]

    def get_embeddings(
        self, encoded_input=None, sents=[], pooling_type="mean", tokenized=True
    ):
        if not tokenized:
            encoded_input = self.tokenizer(
                sents, padding=True, truncation=True, return_tensors="pt"
            )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        if pooling_type == "mean":
            sentence_embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
        elif pooling_type == "max":
            sentence_embeddings = self.max_pooling(
                model_output, encoded_input["attention_mask"]
            )
        elif pooling_type == "cls":
            sentence_embeddings = self.cls_pooling(model_output)
        else:
            print("Pooling type unsupported")
            return None
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.to(device)


def get_similar(embeddings, query_embedding, top_k=10):
    sims = np.dot(query_embedding, embeddings.T)
    most_similar_indices = np.argsort(sims)[::-1][:top_k]
    most_similar_scores = np.sort(sims)[::-1][:top_k]
    return most_similar_indices, most_similar_scores
