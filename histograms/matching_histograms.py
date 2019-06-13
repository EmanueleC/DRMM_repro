import numpy as np
import numba as nb
import math

space = np.linspace(-1, 1, 30)


@nb.jit(nopython=True, fastmath=True)
def nb_cosine(x, y):
    xx, yy, xy = 0.0, 0.0, 0.0
    for i in range(len(x)):
        xx += x[i] * x[i]
        yy += y[i] * y[i]
        xy += x[i] * y[i]
    score = xy / np.sqrt(xx * yy)
    if score >= 1.0:
        index = 29
    else:
        index = np.searchsorted(space, score, side='right')
    return index


class MatchingHistograms:

    def __init__(self, num_bins, max_query_len):
        self.num_bins = num_bins
        self.max_query_len = max_query_len

    def make_histogram(self, query_term, doc, corpus_model, outv):
        matching_histogram = [0] * self.num_bins
        qtv = corpus_model[query_term]
        for doc_term in doc:
            index = nb_cosine(qtv, corpus_model[doc_term])
            matching_histogram[index] = matching_histogram[index] + 1
        return matching_histogram

    def get_histograms(self, query, document, corpus_model, outv, oov_query, oov_document, histograms_mode):
        histograms = []
        for query_term in query:
            histogram = self.make_histogram(query_term, document, corpus_model, outv)
            histograms.append(histogram)
        for query_term in oov_query:
            matching_histogram = [0] * self.num_bins
            count = sum([1 for document_term in oov_document if document_term == query_term])
            matching_histogram[29] = count
            histograms.append(matching_histogram)
        for _ in range(self.max_query_len - len(histograms)):  # padding histogram
            histograms.append([0] * self.num_bins)
        if histograms_mode == "lch":
            histograms = self.norm_hist_log(histograms)
        elif histograms_mode == "nh":
            histograms = self.norm_hist_sum(histograms)
        return histograms

    @staticmethod
    def norm_hist_sum(histograms):
        """
        :param histograms: count-based histograms dictionary
        :return: histograms dictionary normalized by sum
        """
        return [[h / sum(l) if sum(l) != 0 else h for h in l] for l in histograms]

    @staticmethod
    def norm_hist_log(histograms):
        """
        :param histograms: count-based histograms dictionary
        :return: histograms dictionary normalized by log
        """
        return [[math.log(h) if h > 0 else h for h in l] for l in histograms]
