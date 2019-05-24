from utilities.utilities import load_from_pickle_file, save_to_pickle_file
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

words = set()
for topic_id, topic in sorted(queries_obj.items()):
    words.update(topic.get_text().split())

idfs = corpus_obj.calculate_idf(words)

assert len(words) == len(idfs.keys())

idf_filename = "preprocessing/pre_data/idfs/idfs" + conf

save_to_pickle_file(idf_filename, idfs)
