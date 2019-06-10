from utilities.utilities import load_from_pickle_file, save_to_pickle_file
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
from tqdm import tqdm
import json

with open('config.json') as config_file:
    data = json.load(config_file)

stopwords = data["stopwords"]
stemmed = data["stemmed"]
conf = data["conf"]

corpus_filename = "preprocessing/pre_data/Corpus/Corpus" + conf
queries_filename = "preprocessing/pre_data/Queries/Queries" + conf

corpus_obj = load_from_pickle_file(corpus_filename)
queries_obj = load_from_pickle_file(queries_filename)

tfidf = TfidfVectorizer()
tfidf.fit(chain((doc.get_text() for doc in tqdm(corpus_obj.docs.values())), (query.get_text() for query in queries_obj.values())))
idfs = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

# assert len(words) == len(idfs.keys())

idf_filename = "preprocessing/pre_data/idfs/idfs" + conf

save_to_pickle_file(idf_filename, idfs)
