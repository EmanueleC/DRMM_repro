from utilities.utilities import load_all_data, load_glove_model
from preprocessing.prepare_ids import *
from tqdm import tqdm
import json


def create_dict(use_glove):
    print("Creating dictionary")
    word_dict = {}
    dct = model
    if not use_glove:
        dct = model.wv.vocab
    for word in tqdm(dct):
        word_dict[word] = len(word_dict)
    return word_dict


def encode(text, word_dict):
    encoded = []
    for word in text.split():
        code = word_dict.get(word)
        if code is not None:
            encoded.append(code)
    return encoded


def encode_oov(text, word_dict):
    return [word for word in text.split() if word not in word_dict]


def encode_we(word_dict, use_glove):
    print("Encoding word embeddings")
    we = {}
    dct = model
    if not use_glove:
        dct = model.wv
    for word in tqdm(word_dict):
            we[word_dict[word]] = dct[word]
    return we


def encode_idf(idfs, word_dict):
    print("Encoding idf")
    idfs_new = {}
    for word, value in tqdm(idfs.items()):
        code = word_dict.get(word)
        if code is not None:
            idfs_new[word_dict[word]] = value
    return idfs_new


with open('config.json') as config_file:
    data = json.load(config_file)

stopwords = data["stopwords"]
stemmed = data["stemmed"]
glv = data["use_glove"]
conf = data["conf"]
retrieval_alg = data["retrieval_alg"]

corpus_obj, queries_obj, qrels, _, model = load_all_data(stopwords, stemmed, retrieval_alg)
if glv:
    model = load_glove_model("data/glove.6B.300d.txt")

encoded_docs = {}
encoded_queries = {}
encoded_docs_oov = {}
encoded_queries_oov = {}
word_dict = create_dict(glv)
count = 0

print("Encoding documents")

for doc_id, doc in tqdm(corpus_obj.docs.items()):
    encoded = encode(doc.get_text(), word_dict)
    if len(encoded) == 0:
        count += 1
    else:
        encoded_docs[doc_id] = encoded
        encoded_docs_oov[doc_id] = encode_oov(doc.get_text(), word_dict)

print("Encoding queries")

for query_id, query in tqdm(queries_obj.items()):
    encoded_queries[query_id] = encode(query.title, word_dict)
    encoded_queries_oov[query_id] = encode_oov(query.title, word_dict)

idf_filename = "preprocessing/pre_data/idfs/idfs" + conf
idfs = load_from_pickle_file(idf_filename)

idfs = encode_idf(idfs, word_dict)

we = encode_we(word_dict, glv)

max_query_len = max([len(q.title.split()) for q in queries_obj.values()])

padded_query_idfs = {}
padded_query_embs = {}

print("Encoding padded queries idf and embeddings")

for query_id, query in tqdm(encoded_queries.items()):  # padding queries idfs and queries embeddings
    padded_query_idfs[query_id] = []
    padded_query_embs[query_id] = []
    for query_term in query:
        padded_query_idfs[query_id].append(idfs[query_term])
        padded_query_embs[query_id].append(we[query_term])
    for _ in range(max_query_len - len(query)):
        padded_query_idfs[query_id].append(0.0)
        padded_query_embs[query_id].append([0] * 300)

save_to_pickle_file("preprocessing/encoded_data/vocabulary/word_index" + conf, word_dict)
save_to_pickle_file("preprocessing/encoded_data/Corpus/Corpus_encoded" + conf, encoded_docs)
save_to_pickle_file("preprocessing/encoded_data/Queries/Queries_encoded" + conf, encoded_queries)
save_to_pickle_file("preprocessing/encoded_data/Corpus/Corpus_encoded_oov" + conf, encoded_docs_oov)
save_to_pickle_file("preprocessing/encoded_data/Queries/Queries_encoded_oov" + conf, encoded_queries_oov)
save_to_pickle_file("preprocessing/encoded_data/preranked/preranked_total", qrels.ground_truth)
save_to_pickle_file("preprocessing/encoded_data/embeddings/word_embeddings" + conf, we)
save_to_pickle_file("preprocessing/encoded_data/idfs/padded_query_idfs" + conf, padded_query_idfs)
save_to_pickle_file("preprocessing/encoded_data/embeddings/padded_query_embs" + conf, padded_query_embs)
