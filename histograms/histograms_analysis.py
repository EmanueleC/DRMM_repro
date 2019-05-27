from utilities.utilities import plot_histograms, load_from_pickle_file
from histograms.matching_histograms import MatchingHistograms
import random
import json

with open('config.json') as config_file:
    data = json.load(config_file)

random.seed(data["seed"])
stopwords = data["stopwords"]
stemmed = data["stemmed"]
histograms_mode = data["hist_mode"]
conf = data["conf"]

queries_filename = "preprocessing/encoded_data/Queries/Queries_encoded" + conf
corpus_filename = "preprocessing/encoded_data/Corpus/Corpus_encoded" + conf
corpus_model_filename = "preprocessing/encoded_data/embeddings/word_embeddings" + conf
oov_queries_filename = "preprocessing/encoded_data/Queries/Queries_encoded_oov" + conf
oov_corpus_filename = "preprocessing/encoded_data/Corpus/Corpus_encoded_oov" + conf
qrels_filename = "preranked/preranked_total"
queries = load_from_pickle_file(queries_filename)
corpus = load_from_pickle_file(corpus_filename)
corpus_model = load_from_pickle_file(corpus_model_filename)
oov_corpus = load_from_pickle_file(oov_corpus_filename)
oov_queries = load_from_pickle_file(oov_queries_filename)
qrels = load_from_pickle_file(qrels_filename)

topic = "301"
max_query_len = len(queries.get(topic))
num_bins = 30
matching_histograms = MatchingHistograms(num_bins, max_query_len)
positive_doc = random.choice(list(qrels.get_relevant_docs(topic).keys()))
negative_doc = random.choice(list(qrels.get_non_relevant_docs(topic).keys()))
query = queries.get(topic)
pos_document = corpus.get(positive_doc[1])
pos_oov_document = oov_corpus.get(positive_doc[1])
neg_document = corpus.get(negative_doc[1])
neg_oov_document = oov_corpus.get(negative_doc[1])
oov_query = oov_queries.get(topic)
pos_hist = matching_histograms.get_histograms(query, pos_document, corpus_model, oov_query, pos_oov_document, histograms_mode)
neg_hist = matching_histograms.get_histograms(query, neg_document, corpus_model, oov_query, neg_oov_document, histograms_mode)
plot_histograms(positive_doc, pos_hist, histograms_mode, "positive", conf)
plot_histograms(negative_doc, neg_hist, histograms_mode, "negative", conf)
