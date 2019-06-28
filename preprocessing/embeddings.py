from utilities.utilities import load_from_pickle_file, embeddings
import json

with open('config.json') as config_file:
    data = json.load(config_file)

stopwords = data["stopwords"]
stemmed = data["stemmed"]
conf = data["conf"]


def text_embeddings(data, file_name, q):

    print("Embeddings... " + file_name)

    algo = 0  # training algorithm: 0 for CBOW, 1 for skip-gram
    size = 300  # size of embeddings (length of word-vectors)
    window = 10  # maximum distance between a target word and words around it
    min_count = 10  # minimum count of words to consider when training the model
    mode = 0  # activation function: 0 for negative sampling, 1 for hierarchical softmax
    negative = 10  # negative samples
    sample = 1e-4  # negative sub-sample for infrequent words

    if q:
        min_count = 1  # in a query all terms should be considered

    # memory required (approx.) #vocabulary * #size * 4 (float)
    embeddings(data, algo, size, window, min_count, mode, negative, sample, file_name + ".bin")


'''corpus_filename = "preprocessing/pre_data/Corpus/Corpus" + conf
queries_filename = "preprocessing/pre_data/Queries/Queries" + conf'''
corpus_model_filename = "preprocessing/pre_data/models/corpus_model" + conf
queries_model_filename = "preprocessing/pre_data/models/queries_model" + conf

corpus_sent_filename = "preprocessing/pre_data/Corpus/sents_corpus" + conf
queries_sent_filename = "preprocessing/pre_data/Queries/sents_queries" + conf

corpus_sent = load_from_pickle_file(corpus_sent_filename)
queries_sent = load_from_pickle_file(queries_sent_filename)

# corpus_obj = load_from_pickle_file(corpus_filename)

text_embeddings(corpus_sent, corpus_model_filename, False)

# queries_obj = load_from_pickle_file(queries_filename)

# lines_queries = [query.get_text().split() for query in queries_obj.values()]

text_embeddings(queries_sent, queries_model_filename, True)
