"""
Adapted from Sentence Bert Fast Clustering method.

Create subset variants for a train dataset with columns - "label" and "sentence"

"""
import argparse
import collections
import os
import re
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize
from torch import Tensor


class Embeddings:
    """
    Class for all creating all embeddings. Currently written for Sentence transformer embeddings
    Returns normalized embeddings
    """

    def __init__(self, model_name=None):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise e

    def sentence_transformer_encode(self, all_queries):
        embeddings = self.model.encode(all_queries, show_progress_bar=False)
        embeddings = normalize(embeddings, axis=0)
        return embeddings

    def encode(self, all_queries):
        embeddings = self.sentence_transformer_encode(all_queries)
        return embeddings


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def cluster_detection(
    embeddings, threshold=0.75, min_cluster_size=10, max_cluster_size=1000
):
    """
    Function for Cluster Detection.
    Adapted from Sentence Bert Fast Clustering method.
    """

    # Compute cosine similarity scores
    cos_scores = cos_sim(embeddings, embeddings)

    # Minimum size for a community
    try:
        top_k_values, _ = cos_scores.topk(k=min_cluster_size, largest=True)
    except:
        return []

    # Filter for rows >= threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(
                k=max_cluster_size, largest=True
            )
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(
        extracted_communities, key=lambda x: len(x), reverse=True
    )

    # Cleaning communities - Method of merging overlapped clusters or creating new clusters as needed
    unique_communities = []
    extracted_ids = set()
    extracted_id_community = {}
    community_members = {}

    for i, community in enumerate(extracted_communities):
        add_cluster = True
        overlapped_ids = []
        new_members = []
        new_community = []
        for idx in community:
            if idx in extracted_ids:
                overlapped_ids.append(idx)
                add_cluster = False
            else:
                new_members.append(idx)
                new_community.append(idx)

        tot_members = overlapped_ids + new_members
        # identify community overlaps - more than 30 % overlap
        if len(overlapped_ids) * 100.0 / len(tot_members) >= 30:
            overlapping_communities = []
            for idx in overlapped_ids:
                old_community = extracted_id_community[idx]
                overlapping_communities.append(old_community)

            counter = collections.Counter(overlapping_communities)
            old_community = counter.most_common(1)
            community_members[old_community[0][0]] = (
                community_members[old_community[0][0]] + new_members
            )
        else:
            add_cluster = True

        if add_cluster:
            community_members[i] = new_community
            for idx in new_community:
                extracted_ids.add(idx)
                extracted_id_community[idx] = i

    unique_communities = [
        community_members[key]
        for key in community_members.keys()
        if len(community_members[key]) >= min_cluster_size
    ]

    return unique_communities


def find_examplars(data_dict, n_examples: int = None):
    """
    Function to find exemplars
    """
    embedding_model = Embeddings(model_name="all-MiniLM-L12-v2")
    subset_data_labels = []
    subset_data_examples = []
    for label in data_dict.keys():
        label_sents = data_dict[label]
        all_queries = label_sents["sentences"].split(";")
        if n_examples and len(all_queries) <= n_examples:
            subset_data_examples = subset_data_examples + all_queries
            subset_data_labels = subset_data_labels + [label] * len(all_queries)
            continue
        else:
            all_queries = np.array(all_queries)
            query_embeddings = embedding_model.encode(all_queries)
            clusters = cluster_detection(
                embeddings=query_embeddings,
                min_cluster_size=1,
                threshold=0.6,
                max_cluster_size=len(all_queries),
            )

            clustering_indx = np.array([-1] * len(all_queries))

            for i, cluster in enumerate(clusters):
                for sentence_id in cluster:
                    clustering_indx[sentence_id] = i

            cluster_nos = []
            query_sents = []

            for each in set(clustering_indx):
                cluster_nos.append(each)
                filtered_index = np.where(clustering_indx == each)[0]
                query_sents.append(list(all_queries[filtered_index]))

            cluster_dict = {
                str(cluster_no): query_sent
                for cluster_no, query_sent in zip(cluster_nos, query_sents)
            }

            reduced_examples = []
            for c in cluster_dict:
                reduced_examples.append(cluster_dict[c][0])
            if len(reduced_examples) < n_examples:
                not_used_set = list(set(all_queries) - set(reduced_examples))
                reduced_examples = (
                    reduced_examples
                    + not_used_set[: n_examples - len(reduced_examples)]
                )
            subset_data_examples = subset_data_examples + reduced_examples
            subset_data_labels = subset_data_labels + [label] * len(reduced_examples)

    subset_data = pd.DataFrame(
        {"label": subset_data_labels, "sentence": subset_data_examples}
    )

    return subset_data


def subset_labels(dataframe_path, train_file_name, n_examples=5):
    data = pd.read_csv(os.path.join(dataframe_path, train_file_name))
    data["sentences"] = data.groupby(["label"])["sentence"].transform(
        lambda x: ";".join(x)
    )
    data = data.drop(["sentence"], axis=1)
    data = data.drop_duplicates()
    data_dict = data.set_index("label").T.to_dict()
    subset_data = find_examplars(data_dict, n_examples=n_examples)
    subset_data.to_csv(os.path.join(dataframe_path, "subset_" + train_file_name))
    print("Subset creation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe_path", type=str, help="path to train dataset")
    parser.add_argument("--train_file_name", type=str, help="train file name")
    parser.add_argument(
        "--n_examples", type=int, default=5, help="minimum number of examples"
    )
    args = parser.parse_args()
    subset_labels(
        dataframe_path=args.dataframe_path,
        train_file_name=args.train_file_name,
        n_examples=args.n_examples,
    )
