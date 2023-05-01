
"""Module for computing performance metrics

"""
import math
import numbers
from pathlib import Path
import ipdb
import numpy as np
import torch
import scipy.stats
from sklearn.metrics import average_precision_score
import ipdb
import pdb
import itertools


def retrieval_metrics(sims, break_ties='optimistically', complete_dataset_size=None):
    num_queries, num_vids = sims.shape
    if complete_dataset_size is not None:
        num_queries = complete_dataset_size

    sx = np.sort(-sims, axis=1)
    d = np.diag(-sims)
    d = d[:, np.newaxis]
    diff = sx - d
    if break_ties == 'optimistically':
        ind = np.argmax(diff == 0, axis=1)
    elif break_ties == 'averaging':
        locs = np.argwhere(diff == 0)
        grouped_locs = [list(values) for n_row, values in itertools.groupby(locs, key=lambda x: x[0])]
        ind = [np.mean(list(map(lambda x: x[1], locs))) for locs in grouped_locs]
        ind = np.array(ind)
    else:
        raise NotImplementedError
    return cols2metrics(ind, num_queries)


class RetrievalMetric:
    def __init__(self, task='t2v', break_ties='optimistically'):
        task = task.replace('_metrics', '')
        self._task = task

        mod1, mod2 = self._task.split('2')
        self._inv_task = f"{mod2}2{mod1}"
        self.__name__ = f"{self._task}_metrics"
        self.break_ties = break_ties

    def __call__(self, sims_dict, complete_dataset_size=None):
        if self._task in sims_dict:
            return retrieval_metrics(sims_dict[self._task], complete_dataset_size=complete_dataset_size)
        elif self._inv_task in sims_dict:
            return retrieval_metrics(sims_dict[self._inv_task].T, complete_dataset_size=complete_dataset_size)
        else:
            return {}

    def __repr__(self):
        return f"{self._task}_metrics"


def retrieval_as_classification(sims, query_masks=None):
    """Compute classification metrics from a similiarity matrix.
    """
    assert sims.ndim == 2, "expected a matrix"

    # switch axes of query-labels and video
    sims = sims.T
    query_masks = query_masks.T
    dists = -sims
    num_queries, num_labels = sims.shape
    break_ties = "averaging"

    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]

        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        # min_rank = np.inf
        label_ranks = []
        for gt_label in np.where(query_masks[ii, :])[0]:
            ranks = np.where((sorted_dists - row_dists[gt_label]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                # NOTE: If there is more than one caption per video, its possible for the
                # method to do "worse than chance" in the degenerate case when all
                # similarities are tied.  TODO(Samuel): Address this case.
                rank = ranks.mean()
            else:
                raise ValueError(f"unknown tie-breaking method: {break_ties}")
            label_ranks.append(rank)
        # Avoid penalising for assigning higher similarity to other gt labels. This is
        # done by subtracting out the better ranked query labels.  Note that this step
        # introduces a slight skew in favour of videos with lots of labels.  We can
        # address this later with a normalisation step if needed.
        label_ranks = [x - idx for idx, x in enumerate(label_ranks)]

        # Include all labels in the final calculation
        query_ranks.extend(label_ranks)
    query_ranks = np.array(query_ranks)

    return cols2metrics(query_ranks, num_queries=len(query_ranks))


def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    # metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries # Don't need R50
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def mean_average_precision(sims, query_masks=None):
    ap_meter = APMeter()
    ap_meter.add(output=sims.T, target=query_masks.T)
    return {"mAP": ap_meter.value().mean()}

def acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def video_precision(output, target):
    """ percentage of videos which have been aligned to a matching text pair"""
    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    correct = 0
    for bout, btarg in zip(output, target):
        for pair in bout:
            eq = torch.eq(pair, btarg)
            if torch.logical_and(eq[:, 0], eq[:, 1]).any():
                correct += 1
    return correct / (target.shape[0] * target.shape[1])

def video_precision_adj(output, target):
    """ adjusts the video precision metric by ignoring videos which have no aligning text."""
    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    correct = 0
    for bout, btarg in zip(output, target):
        for pair in bout:
            eq = torch.eq(pair, btarg)
            if torch.logical_and(eq[:, 0], eq[:, 1]).any():
                correct += 1
    denom = len(target[:, :, 0].unique())

    return correct / denom

def video_precision_adj(output, target):
    """ adjusts the video precision metric by ignoring videos which have no aligning text."""
    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    correct = 0
    for bout, btarg in zip(output, target):
        for pair in bout:
            eq = torch.eq(pair, btarg)
            if torch.logical_and(eq[:, 0], eq[:, 1]).any():
                correct += 1
    denom = len(target[:, :, 0].unique())

    return correct / denom