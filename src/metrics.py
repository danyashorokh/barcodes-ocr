
import numpy as np

from textdistance import levenshtein


def accuracy(y_true, y_pred):
    """Calc accuracy between two list of strings."""
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score = np.mean(scores)
    return avg_score


def cer(gt_texts, pred_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_chars = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        lev_distances += levenshtein.distance(pred_text, gt_text)
        num_gt_chars += len(gt_text)
    return lev_distances / num_gt_chars


def wer(gt_texts, pred_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_words = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        gt_words, pred_words = gt_text.split(), pred_text.split()
        lev_distances += levenshtein.distance(pred_words, gt_words)
        num_gt_words += len(gt_words)
    return lev_distances / num_gt_words
