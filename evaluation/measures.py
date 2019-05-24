from utils.utils import *
import math

# Evaluation measures for a run


def precision(y_true, cutoff):
    """
    :param y_pred: predictions / score of the run
    :param y_true: ground truth values of the run
    :param cutoff: cutoff evaluation level
    :return: precision at cutoff k (prec@k)
    """
    y_true = y_true[:cutoff+1]
    rel = 0
    if cutoff <= len(y_true):
        total = len(y_true)
    else:
        total = cutoff
    for el in y_true:
        if el[0] > 0:
            rel = rel + 1
    return rel/total


def average_precision(y_true, total_rel):
    """
    :param y_pred: predictions / score of the run
    :param y_true: ground truth values of the run
    :return: average precision
    """
    sum_prec_weigth = 0
    for i in range(0, len(y_true)):
        if y_true[i] > 0:
            sum_prec_weigth = sum_prec_weigth + precision(y_true, i)
    if total_rel == 0:
        return 0
    else:
        return sum_prec_weigth / total_rel


def map_measure(runs):
    """
    :param runs: run for a topic obtained by a retrieval system
    :return: mean average precision
    """
    if len(runs) <= 0:
        return 0
    else:
        sum_avgprec_q = 0
        for run in runs:
            sum_avgprec_q = sum_avgprec_q + run.get_average_precision()
        return sum_avgprec_q / len(runs)


def dcg(cutoff, scores):
    """
    :param cutoff: cutoff evaluation level
    :param scores: predictions / score of the run
    :return: discounted cumulative gain at cutoff k (dcg@k)
    """
    acc = 0
    up = cutoff
    if len(scores) < cutoff:
        up = len(scores)
    for i in range(0, up):
        acc = acc + scores[i] / math.log(i + 2, 2)
    return acc


def ndcg(cutoff, y_pred, ideal):
    """
    :param cutoff: cutoff evaluation level
    :param y_pred: predictions / score of the run
    :param y_true: ground truth values of the run
    :return: normalized discounted cumulative gain at cutoff k (ndcg@k)
    """
    ideal.sort()
    ideal = ideal[::-1]  # ideal run
    ideal_dcg = dcg(cutoff, ideal)
    if ideal_dcg != 0:
        return dcg(cutoff, y_pred) / ideal_dcg
    else:
        return 0


def fall_out(y_true, total_nonrel):
    """
    :param y_pred: predictions / score of the run
    :param y_true: ground truth values of the run
    :param total_nonrel: total (true) non relevant document in the whole run topic
    :return: fallout for a run (probability that a non-relevant document is
    retrieved by the query)
    """
    count_nonrel_retrieved = 0
    for el in y_true:
        if el == 0:
            count_nonrel_retrieved = count_nonrel_retrieved + 1
    if total_nonrel == 0:
        return 0
    else:
        return count_nonrel_retrieved / total_nonrel


def mean_reciprocal_rank(y_true, cutoff):
    curr_sum = 0
    if cutoff > len(y_true):
        cutoff = len(y_true)
    for i in range(0, cutoff):
        if y_true[i] > 0:
            curr_sum = curr_sum + 1 / (len(y_true) - i)
    return curr_sum / len(y_true)
