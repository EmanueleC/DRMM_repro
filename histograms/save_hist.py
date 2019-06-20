from histograms.matching_histograms import MatchingHistograms
from utilities.utilities import load_from_pickle_file, save_to_pickle_file, load_glove_model
from tqdm import tqdm
import json


def save_local_histograms(retrieval_alg, outv, num_bins, max_query_len, conf, histograms_mode):
    preranked_total_filename = "preranked/preranked_total_" + retrieval_alg
    preranked_total = load_from_pickle_file(preranked_total_filename)
    histograms = {}
    matching_histograms = MatchingHistograms(num_bins, max_query_len)
    for (query_id, document_id) in tqdm(preranked_total.ground_truth.keys()):
        query = queries.get(query_id)
        document = corpus.get(document_id)
        if query is not None and document is not None:
            oov_document = oov_corpus.get(document_id)
            oov_query = oov_queries.get(query_id)
            hist = matching_histograms.get_histograms(query, document, model, outv, oov_query, oov_document, histograms_mode)
            histograms[(query_id, document_id)] = hist
    save_to_pickle_file("preprocessing/encoded_data/histograms/histograms_total" + conf + "_" + histograms_mode, histograms)


with open('config.json') as config_file:
    data = json.load(config_file)

stopwords = data["stopwords"]
stemmed = data["stemmed"]
histograms_mode = data["hist_mode"]
glv = data["use_glove"]
conf = data["conf"]
retrieval_alg = data["retrieval_alg"]

num_bins = 30

queries_filename = "preprocessing/encoded_data/Queries/Queries_encoded" + conf
corpus_filename = "preprocessing/encoded_data/Corpus/Corpus_encoded" + conf
corpus_model_filename = "preprocessing/encoded_data/embeddings/word_embeddings" + conf
corpus_model_out_filename = "preprocessing/encoded_data/embeddings/word_embeddings_out" + conf
oov_queries_filename = "preprocessing/encoded_data/Queries/Queries_encoded_oov" + conf
oov_corpus_filename = "preprocessing/encoded_data/Corpus/Corpus_encoded_oov" + conf
queries = load_from_pickle_file(queries_filename)
corpus = load_from_pickle_file(corpus_filename)
model = load_from_pickle_file(corpus_model_filename)
model_out = load_from_pickle_file(corpus_model_out_filename)
oov_corpus = load_from_pickle_file(oov_corpus_filename)
oov_queries = load_from_pickle_file(oov_queries_filename)

max_query_len = max([len(q) for q in queries.values()])

save_local_histograms(retrieval_alg, model_out, num_bins, max_query_len, conf + "_glove_" + str(glv), histograms_mode)
