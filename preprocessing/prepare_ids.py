from utilities.utilities import load_from_pickle_file, save_to_pickle_file
from tqdm import tqdm
import random

labels_train_filename = "preprocessing/pre_data/ids/ids_train"
labels_validation_filename = "preprocessing/pre_data/ids/ids_validation"
labels_test_filename = "preprocessing/pre_data/ids/ids_test"

n_pos = 50
n_neg = 50
n_val = 10
k = 5  # for k-fold cross validation
random.seed(42)


def prepare_train_validation_ids(qrels, topics_train):

    ids_train = []
    ids_validation = []
    for topic in tqdm(topics_train):
        rels = list(qrels.get_relevant_docs(str(topic)).keys())
        nonrels = list(qrels.get_non_relevant_docs(str(topic)).keys())
        # possible repetition
        if len(rels) > 0 and len(nonrels) > 0:
            i = 0
            j = 0
            while i < n_pos + n_neg:
                ids_train.append(random.choice(rels))
                ids_train.append(random.choice(nonrels))
                i += 2
            while j < n_val:
                ids_validation.append(random.choice(rels))
                ids_validation.append(random.choice(nonrels))
                j += 2
    #("len training labels", len(ids_train))
    #print("len validation labels", len(ids_validation))
    return ids_train, ids_validation


def prepare_test_ids(qrels, topics_test):
    ids_test = {}
    for topic in topics_test:
        pairs = qrels.get_pairs_topic(str(topic))
        ids_test.update(pairs)
    print("len test labels", len(ids_test.keys()))
    return list(ids_test.keys())


if __name__ == "__main__":

    qrels_filename = "preranked/preranked_total"

    qrels = load_from_pickle_file(qrels_filename)

    topics = list(range(301, 451)) + list(range(601, 701))
    random.shuffle(topics)

    cleared_ids_train = []
    cleared_ids_validation = []
    cleared_ids_test = []

    for i in range(k):
        topic_test = topics[i*50:(i+1)*50]
        topic_train = [topic for topic in topics if topic not in topic_test]
        ids_train, ids_validation = prepare_train_validation_ids(qrels, topic_train)
        ids_test = prepare_test_ids(qrels, topic_test)
        cleared_ids_train.append(ids_train)
        cleared_ids_validation.append(ids_validation)
        cleared_ids_test.append(ids_test)

    save_to_pickle_file("preprocessing/encoded_data/ids/cleared_ids_train", cleared_ids_train)
    save_to_pickle_file("preprocessing/encoded_data/ids/cleared_ids_validation", cleared_ids_validation)
    save_to_pickle_file("preprocessing/encoded_data/ids/cleared_ids_test", cleared_ids_test)
