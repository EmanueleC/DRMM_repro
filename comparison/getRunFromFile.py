from utilities.utilities import Qrels, load_from_pickle_file
from tqdm import tqdm
import pickle
import json

qrels_file = open('preprocessing/pre_data/Qrels/Qrels_cleaned', 'rb')
qrels_obj = pickle.load(qrels_file)
qrels_file.close()

qrels_file = load_from_pickle_file('preprocessing/pre_data/Qrels/Qrels_cleaned')
corpus_obj = load_from_pickle_file('preprocessing/pre_data/Corpus/Corpus')
queries_obj = load_from_pickle_file('preprocessing/pre_data/Queries/Queries')

num_topics = len(queries_obj)

with open('config.json') as config_file:
    data = json.load(config_file)

retrieval_alg = data["retrieval_alg"]

if retrieval_alg == "QL":
    preranked_filename = "comparison/terrier_preranked/DirichletLM_6.res"
elif retrieval_alg == "Bm25":
    preranked_filename = "comparison/terrier_preranked/BM25.res"

""" create runs objects from galago batch-search output """
with open(preranked_filename, 'r') as results:
    runsList = (line.split() for line in results)

    runs = {}
    sum = 0
    for i in tqdm(range(num_topics)):
        if i == 0:
            line = next(runsList)
        scores = []
        topic_id = line[0]
        keys_qrels = qrels_obj.get_pairs()
        ranklist = {}
        while line[0] == topic_id:
            doc_id = line[2]
            if (topic_id, doc_id) in keys_qrels:
                score = qrels_obj.ground_truth[(topic_id, doc_id)]
                if score > 1:
                    score = 1  # all binary scores
            else:
                score = 0
            scores.append(score)
            if queries_obj.get(topic_id) is not None and corpus_obj.docs.get(doc_id) is not None:
                ranklist[doc_id] = score
            try:
                line = next(runsList)
            except StopIteration:
                line = "0"
        if len(scores) > 0:
            print(topic_id, len(scores))
            count = len(list(ranklist.keys()))
            sum = sum + count
            print(topic_id, count)
            runs[topic_id] = ranklist
        else:
            print(topic_id, len(scores))
    print(sum)

    file_preranked = open("preranked/preranked_" + retrieval_alg + ".txt", "w")

    preranked_total = Qrels()
    text = ""
    for topic_id, ranklist in tqdm(runs.items()):
        i = 0
        for doc, score in ranklist.items():
            preranked_total.add_entry(topic_id, doc, score)
            text = text + str(topic_id) + " Q0 " + str(doc) + " " + str(i) + " " + str(score) + " DRMM_PRE_Q\n"
            i = i + 1
    file_preranked.write(text)

    with open("preranked/preranked_total_" + retrieval_alg, 'wb') as pickle_file:
        pickle.dump(preranked_total, pickle_file)
