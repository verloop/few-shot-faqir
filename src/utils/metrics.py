import logging
import math
from typing import List

import numpy as np
from sklearn.metrics import f1_score


def compute_v_batch(actual, predicted, k):
    """
    Computes whether predicted value is present in actual and assigns either a 1 or a 0

    v = np.array([[1,0],[1,0]])
    """
    v = []
    for i, p_i in enumerate(predicted):
        v_i = []
        if len(p_i) < k:
            logging.warning(
                "Cannot compute at k since prediction doesnt have enough values"
            )
            raise
        for j, p in enumerate(p_i[:k]):
            if p in actual[i] and p not in p_i[:j]:
                v_i.append(1)
            else:
                v_i.append(0)
        v.append(v_i)
    v = np.array(v)
    return v


def precision_at_k_batch(actual: List[List], predicted: List[List], k: int) -> float:
    """
    Computes the precision at k

    Arguments
    --------
    actual : list of a list of FAQ correct responses for all user utterances [[a1,a2],[a6,a8]..]
    Each sublist corresponds to correct FAQ responses for one utterance

    predicted : list of a list of predicted responses [[a1,a5],[a8,a9]..] ordered by scores
    k : maximum number of elements to be considered for each user utterance

    Returns:
    --------
    Precision @ k for the batch
    """
    v = compute_v_batch(actual, predicted, k)
    if k > len(predicted[0]):
        k = len(predicted[0])
    tot_correct = np.sum(v[:, :k])
    n = v.shape[0]
    precision = float(tot_correct) / (n * k)
    return precision


def precision_at_k_single(actual: List, predicted: List, k: int) -> float:
    """
    Computes the precision for a single utterance given correct FAQ responses and predicted FAQ responses

    Arguments
    ----------
    actual : List of valid FAQ responses for a user utterance
    predicted : List of predicted FAQ responses for a user utterance ordered by scores

    k : maximum number of elements to be considered for each user utterance

    Returns
    -------
    precision@k

    """
    correct = 0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            correct += 1
    return correct / min(len(predicted), k)


def success_rate_at_k_batch(actual: List[List], predicted: List[List], k: int) -> float:
    """
    Computes the Success Rate at k
    Success Rate is the fraction of questions for which at least one related question is ranked among the top k

    Arguments
    --------
    actual : list of a list of FAQ correct responses for all user utterances [[a1,a2],[a6,a8]..]
    Each sublist corresponds to correct FAQ responses for one utterance

    predicted : list of a list of predicted responses [[a1,a5],[a8,a9]..] ordered by scores

    k : maximum number of elements to be considered for each user utterance

    Returns:
    --------
    Success Rate @ k for the batch
    """
    v = compute_v_batch(actual, predicted, k)
    if k > len(predicted[0]):
        k = len(predicted[0])
    x = np.sum(v[:, :k], axis=1) > 0
    x = x.astype(int)
    n = v.shape[0]
    tot_correct = np.sum(x)
    sr = float(tot_correct) / n
    return sr


def ap_at_k_single(actual: List, predicted: List, k: int) -> float:
    """
    Computes the average precision at k for one utterance

    Arguments
    ----------
    actual : List of valid FAQ responses for a user utterance
    predicted : List of predicted FAQ responses for a user utterance ordered by scores

    k : maximum number of elements to be considered for each user utterance

    Returns
    -------
    ap : The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    cum_precision_at_k = 0.0
    num_hits = 0.0

    # same as computing cumulative precision_at_k_single for all k from 1 to k where a relevant FAQ response was given
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            cum_precision_at_k += num_hits / (i + 1.0)

    # AP is calculated at cumulative precision for all k / no of relevant docs
    ap = cum_precision_at_k / min(len(actual), k)
    return ap


def map_at_k(actual: List[List], predicted: List[List], k: int) -> float:
    """
    Computes mean average precision of a list of list of items

    Arguments:
    ----------
    actual : list of a list of FAQ correct responses for all user utterances [[a1,a2],[a6,a8]..]
    Each sublist corresponds to correct FAQ responses for one utterance

    predicted : list of a list of predicted responses [[a1,a5],[a8,a9]..] ordered by scores

    k : maximum number of elements to be considered for each user utterance

    Returns :
    ---------
    Mean average precision for multiple queries

    """
    mean_ap = np.mean([ap_at_k_single(a, p, k) for a, p in zip(actual, predicted)])
    return round(mean_ap, 2)


def reciprocal_rank_single(actual: List, predicted: List) -> float:
    """
    Computes Mean Reciprocal Rank based on the first relevant FAQ

    Arguments:
    ---------
    actual : List of valid FAQ responses for a user utterance
    predicted : List of predicted FAQ responses for a user utterance.Order is important

    Returns
    -------
    Reciprocal rank score (1/(rank position of the first relevant FAQ))
    """
    reciprocal_rank_score = 1e-6
    for i, p in enumerate(predicted):
        if p in actual:
            reciprocal_rank_score = 1.0 / (i + 1)
            return reciprocal_rank_score
    return reciprocal_rank_score


def mrr(actual: List[List], predicted: List[List]) -> float:
    mrr = np.mean([reciprocal_rank_single(a, p) for a, p in zip(actual, predicted)])
    return mrr


def dcg_at_k(relevance_list: List, k: int) -> float:
    """
    Discounted Cumulative Gain (DCG)

    Arguments:
    ----------
    relevance_list - a list of elements with relevance Ex: [5,4,2,2,1]
    Returns
    -------
    Discounted Cumulative Gain at rank k

    """
    dcg = 0.0
    for i, val in enumerate(relevance_list[:k]):
        if i == 0:
            dcg += float(val)
        else:
            dcg += float(val) / math.log((i - 1 + 2), 2)
    return dcg


def ndcg_at_k_batch(
    actual: List[List], predicted: List[List], k: int, relevance_scores=False
) -> float:
    """
    Computes normalized discounted cumulative gain (NDCG) for multiple queries by taking a mean

    Arguments:
    ----------
    actual : List of valid FAQ responses for a user utterance
    predicted : List of predicted FAQ responses for a user utterance.Order is important


    Returns
    -------
    Normalized discounted cumulative gain (NDCG) at rank k

    """
    v = compute_v_batch(actual, predicted, k)
    if not relevance_scores:
        relevance = np.zeros_like(v)
        vals = [len(each) for each in actual]
        for i, val in enumerate(vals):
            relevance[i, :val] = 1
        mean_ndcg = np.mean(
            [
                dcg_at_k(v[i], k) / dcg_at_k(relevance[i], k)
                for i in range(0, len(actual))
            ]
        )
    else:
        mean_ndcg = np.mean(
            [dcg_at_k(v[i], k) / dcg_at_k(actual[i], k) for i in range(0, len(actual))]
        )
    return mean_ndcg


def f1_score_k(actual, predicted, k, f1_type="macro"):
    """
    Computes F1 score.
    actual : List of list valid FAQ responses for a user utterance
    predicted : List of list of predicted FAQ responses for a user utterance
    k : maximum number of elements to be considered for each user utterance
    type: macro/micro/weighted
    """
    v = compute_v_batch(actual, predicted, k)
    v_pred = np.sum(v[:, :k], axis=1)
    v_predicted = v_pred > 0
    v_predicted = v_predicted.astype(int)
    v_actual = [1] * len(actual)
    return round(f1_score(v_actual, v_predicted, average=f1_type), 2)
