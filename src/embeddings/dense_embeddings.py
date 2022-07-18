""" Embedding approaches """

import string

import fasttext
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class DenseEmbeddings:
    def __init__(self, model_name="bert-base-uncased", device="cuda"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
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

        return sentence_embeddings.to(self.device)


class FasttextEmbeddings:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.emb_model = fasttext.load_model(self.model_path)

    def get_embeddings(self, sents=[]):
        embeddings = []
        for sent in sents:
            sent = sent.replace("\n", "")
            sent_emb = self.emb_model.get_sentence_vector(sent)
            embeddings.append(sent_emb)
        embeddings = np.array(embeddings)
        return embeddings


class GlovetEmbeddings:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.vocab_embeddings = {}

        with open(self.model_path, "rt") as fi:
            full_content = fi.read().strip().split("\n")

        for i in range(len(full_content)):
            i_word = full_content[i].split(" ")[0]
            i_embeddings = [float(val) for val in full_content[i].split(" ")[1:]]
            self.vocab_embeddings[i_word] = i_embeddings
        self.dim = len(i_embeddings)

    def get_embeddings(self, sents=[]):
        embeddings = []
        for sent in sents:
            sent = sent.lower()
            sent = sent.replace("\n", "")
            sent = sent.translate(str.maketrans("", "", string.punctuation))
            all_word_embs = []
            words = sent.split(" ")
            for word in words:
                if word in self.vocab_embeddings:
                    all_word_embs.append(self.vocab_embeddings[word])
            all_word_embs = np.array(all_word_embs)
            if len(all_word_embs) > 0:
                sent_emb = np.mean(all_word_embs, axis=0)
            else:
                sent_emb = np.zeros(self.dim)
            embeddings.append(list(sent_emb))
        embeddings = np.array(embeddings)
        return embeddings


def get_similar(embeddings, query_embedding, top_k=10):
    sims = np.dot(query_embedding, embeddings.T)
    most_similar_indices = np.argsort(sims)[::-1][:top_k]
    most_similar_scores = np.sort(sims)[::-1][:top_k]
    return most_similar_indices, most_similar_scores
