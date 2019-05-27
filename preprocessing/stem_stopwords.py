from utilities.utilities import load_from_pickle_file, save_to_pickle_file
from tqdm import tqdm
from nltk.stem.porter import *
import json

with open('config.json') as config_file:
    data = json.load(config_file)

stopwords = data["stopwords"]
stemmed = data["stemmed"]
conf = data["conf"]

corpus_filename = "preprocessing/pre_data/Corpus/Corpus"
queries_filename = "preprocessing/pre_data/Queries/Queries"

corpus_obj = load_from_pickle_file(corpus_filename)
queries_obj = load_from_pickle_file(queries_filename)

if stopwords:

    print("Removing stopwords...")

    stopwords_list = []

    with open("inquery", 'r') as f:
        for line in f.readlines():
            stopwords_list.append(line.strip('\n'))

    for _, doc in tqdm(corpus_obj.docs.items()):
        doc.headline = " ".join([word for word in doc.headline.split() if word not in stopwords_list])
        doc.content = " ".join([word for word in doc.content.split() if word not in stopwords_list])

    for _, query in tqdm(queries_obj.items()):
        query.title = " ".join([word for word in query.title.split() if word not in stopwords_list])
        query.desc = " ".join([word for word in query.desc.split() if word not in stopwords_list])

if stemmed:

    stemmer = PorterStemmer()

    vocab = load_from_pickle_file("preprocessing/pre_data/vocabulary")

    for _, query in tqdm(queries_obj.items()):
        vocab.update(query.title.split())
        vocab.update(query.desc.split())

    mapping_stemmed = {}

    print("Stemming...")

    for word in tqdm(vocab):
        mapping_stemmed[word] = stemmer.stem(word)

    for _, doc in tqdm(corpus_obj.docs.items()):
        doc.headline = " ".join([mapping_stemmed[word] for word in doc.headline.split()])
        doc.content = " ".join([mapping_stemmed[word] for word in doc.content.split()])

    for _, query in tqdm(queries_obj.items()):
        query.title = " ".join([mapping_stemmed[word] for word in query.title.split()])
        query.desc = " ".join([mapping_stemmed[word] for word in query.desc.split()])

corpus_filename = "preprocessing/pre_data/Corpus/Corpus"

save_to_pickle_file(corpus_filename + conf, corpus_obj)

stemmed_filename = "preprocessing/pre_data/Corpus/stem_map"

save_to_pickle_file(stemmed_filename + conf, mapping_stemmed)

queries_filename = "preprocessing/pre_data/Queries/Queries"

save_to_pickle_file(queries_filename + conf, queries_obj)
