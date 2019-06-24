from utilities.utilities import Corpus, Qrels, parse_docs, parse_query, save_to_pickle_file, fix_topics
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import re

qrelPath = "data/qrels.robust2004.txt"
topic_path = "data/trec.robust.2004.txt"


def process_file(xml, topic):

    xml = "<?xml version='1.0' encoding='utf8'?>\n" + '<root>' + xml + '</root>'  # Add a root tag
    root = BeautifulSoup(xml, "xml")

    if topic:
        plain_text, obj = parse_query(root, "title")
    else:
        plain_text, obj = parse_docs(root)

    return plain_text, obj


path_files_list = {}
vocabulary = set()

# ignores files not in corpus
for path, subdirs, files in os.walk("data/TIPSTER"):
    if re.search("DTDS", path) is None and re.search("__MACOSX", path) is None and re.search("AUX", path) is None:
        for name in files:
            if re.search(".Z", name) is None and re.search("READ", name) is None and re.search(".DS_Store", name) is None:
                path_files_list[name] = os.path.join(path, name)

corpus_length = len(path_files_list)
print(corpus_length, "files to be processed")
corpus = Corpus()

sents_corpus = []

for fileName in tqdm(list(path_files_list.keys())):

    with open(path_files_list[fileName], 'r', encoding='utf-8', errors='ignore') as f:
        xml = f.read()

    sents_text, corpus_doc = process_file(xml, False)  # all documents with content length == 0 are ignored
    sents_corpus = sents_corpus + sents_text
    vocabulary.update([w for sent in sents_text for w in sent.split()])
    corpus.update(corpus_doc)

# Save corpus object
save_to_pickle_file("preprocessing/pre_data/Corpus/Corpus", corpus)

with open(topic_path, 'r') as f:
    xml = f.read()

xml = fix_topics(xml)

sents_queries, queries = process_file(xml, True)

save_to_pickle_file("preprocessing/pre_data/Corpus/sents_corpus", sents_corpus)
save_to_pickle_file("preprocessing/pre_data/Queries/sents_queries", sents_queries)

# Save Queries object
save_to_pickle_file("preprocessing/pre_data/Queries/Queries", queries)

# Save vocabulary object
save_to_pickle_file("preprocessing/pre_data/vocabulary", vocabulary)

qrels = Qrels()

file_qrels_cleaned = open("preprocessing/pre_data/Qrels/Qrels_cleaned.txt", "w")  # file to be written

with open(qrelPath, 'r') as f:
    for line in f.readlines():
        values = line.split()
        topic_id = values[0]
        doc_id = values[2]
        relevance = values[3]
        if doc_id in corpus.docs:  # clear qrels from docs not in the collection
            qrels.add_entry(topic_id, doc_id, relevance)
            file_qrels_cleaned.write(line)

# Save Qrels object (judgments)
save_to_pickle_file("preprocessing/pre_data/Qrels/Qrels_cleaned", qrels)
