from utilities.utilities import load_from_pickle_file, save_to_pickle_file
from tqdm import tqdm
import random
import itertools
import json

labels_train_filename = "preprocessing/pre_data/ids/ids_train"
labels_test_filename = "preprocessing/pre_data/ids/ids_test"

with open('config.json') as config_file:
    data = json.load(config_file)

n_pos = data["pos"]
n_neg = data["neg"]
k = 5  # for k-fold cross validation
retrieval_alg = data["retrieval_alg"]

random.seed(data["seed"])


def prepare_train_ids(qrels, topics_train, n_pos, n_neg):

    ids_train = []
    for topic in tqdm(topics_train):
        rels = list(qrels.get_relevant_docs(str(topic)).keys())
        nonrels = list(qrels.get_non_relevant_docs(str(topic)).keys())
        pos_train = []
        neg_train = []
        # possible repetition
        if len(rels) > 0 and len(nonrels) > 0:
            i = 0

            while i < n_pos:
                pos_train.append(random.choice(rels))
                i += 1
            i = 0

            while i < n_neg:
                neg_train.append(random.choice(nonrels))
                i += 1

            ids_train += [x for x in itertools.chain.from_iterable(itertools.zip_longest(pos_train, neg_train)) if x]
        else:
            print("Topic empty qrels:", topic)

    print("len training labels", len(ids_train))
    return ids_train


def prepare_test_ids(qrels, topics_test):
    ids_test = {}
    for topic in topics_test:
        pairs = qrels.get_pairs_topic(str(topic))
        ids_test.update(pairs)
    print("len test labels", len(ids_test.keys()))
    return list(ids_test.keys())


if __name__ == "__main__":

    qrels_filename = "preranked/preranked_total_" + retrieval_alg

    qrels = load_from_pickle_file(qrels_filename)

    topics = list(range(301, 451)) + list(range(601, 701))
    random.shuffle(topics)

    cleared_ids_train = []
    cleared_ids_test = []

    for i in range(k):
        topic_test = topics[i*50:(i+1)*50]
        topic_train = [topic for topic in topics if topic not in topic_test]
        ids_train = prepare_train_ids(qrels, topic_train, n_pos, n_neg)
        ids_test = prepare_test_ids(qrels, topic_test)
        cleared_ids_train.append(ids_train)
        cleared_ids_test.append(ids_test)

    save_to_pickle_file("preprocessing/encoded_data/ids/cleared_ids_train_" + retrieval_alg + "_" + str(n_pos) + "_" + str(n_neg), cleared_ids_train)
    save_to_pickle_file("preprocessing/encoded_data/ids/cleared_ids_test_" + retrieval_alg, cleared_ids_test)
