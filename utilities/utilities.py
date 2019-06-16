from itertools import groupby
from typing import Optional, List, Dict, Tuple
from krovetzstemmer import Stemmer
from numba import jit
from gensim.models import Word2Vec
import pickle
import subprocess
import re
import html
import logging
import matplotlib.pyplot as plt
import numpy as np


class Query:
    """
    Query class representation (topic id, title, description)
    data input should be already parsed appropriately and cleaned
    """

    def __init__(self, qid: str, title: str, desc: str) -> None:
        """ initializes query instance
        :param qid: topic id
        :param title: topic title
        :param desc: topic description
        """
        self.qId = qid
        self.title = title
        self.desc = desc

    def get_text(self) -> str:
        """
        :return: topic title and description concatenated
        """
        return ' ' + self.title + ' ' + self.desc + ' '

    def print_query(self) -> None:
        """ Print query attributes
        :return: void
        """
        print(self.qId, self.title, self.desc)

    def get_lines(self) -> List[str]:
        """
        :return: a list with all lines of query text
        """
        return self.get_text().split('\n')


class Document:
    """
    Document class representation (document id + headline + content)
    data input should be already parsed appropriately and cleaned
    """

    def __init__(self, document_id: str, headline: str, content: str) -> None:
        """ initializes document instance
        :param document_id: document id
        :param headline: document headline
        :param content: document content (in <TEXT> tag)
        """
        self.document_id = document_id
        self.headline = headline
        self.content = content

    def get_text(self) -> str:
        """
        :return: document headline and content concatenated
        """
        return ' ' + self.headline + ' ' + self.content + ' '

    def print_doc(self) -> None:
        """ print document attributes
        :return: void
        """
        print(self.document_id, self.headline, self.content)

    def get_lines(self) -> List[str]:
        """
        :return: a list with all lines of document text
        """
        return self.get_text().split('\n')


class Corpus:

    def __init__(self):
        self.docs = {}

    def add_doc(self, doc: Document) -> None:
        self.docs.update({doc.document_id: doc})

    def get_doc(self, doc_id: str) -> Optional[Document]:
        if doc_id in self.docs:
            return self.docs[doc_id]
        else:
            return None

    def get_text(self) -> str:
        text = ""
        for key in self.docs.keys():
            text = text + self.docs[key].get_text()
        return text

    def update(self, corpus: 'Corpus') -> None:
        self.docs.update(corpus.docs)

    '''def get_lines_docs(self) -> List[str]:
        lines_docs = (self.docs[key].get_lines() for key in self.docs.keys())
        return lines_docs'''

    def __iter__(self) -> str:
        for doc in self.docs.values():
            yield doc.get_text().split()


class Qrels:

    def __init__(self) -> None:
        self.ground_truth = dict()

    def add_entry(self, topic: str, document: str, rel: str) -> None:
        self.ground_truth.update({
            (topic, document): float(rel)
        })

    def get_topics(self):
        return set([topic for topic, _ in self.ground_truth.keys()])

    def get_pairs(self):
        return self.ground_truth.keys()

    def get_pairs_topic(self, topic: str) -> Dict[Tuple[str, str], int]:
        pairs_topic = {key: value for key, value in self.ground_truth.items() if key[0] == topic}
        return pairs_topic

    def get_relevant_docs(self, topic: str):
        pair_rel = {key: value for key, value in self.get_pairs_topic(topic).items() if value > 0}
        return pair_rel

    def get_non_relevant_docs(self, topic: str):
        pair_non_rel = {key: value for key, value in self.get_pairs_topic(topic).items() if value == 0}
        return pair_non_rel

    def get_info_topic(self, topic: str) -> Dict[str, int]:
        count_rel = 0
        count_non_rel = 0
        for key in self.ground_truth:
            if key[0] == topic:
                if self.ground_truth[key] > 0:
                    count_rel = count_rel + 1
                else:
                    count_non_rel = count_non_rel + 1
        return {"total_rel": count_rel, "total_nonrel": count_non_rel}

    def get_score(self, topic_id: int, document_id: int) -> int:
        return self.ground_truth[(topic_id, document_id)]


def clear_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub('\n+', ' ', text)
    # text = re.sub('\. ', '\n', text)  # full stop ends a sentence
    text = re.sub(r'-', ' ', text)  # split words between hyphens
    text = re.sub(r'[^\w\s]', '', text)  # keeps any alphanumeric character and the underscore, remove enything else; remove any whitespace
    text = re.sub(r'_', ' ', text)  # remove underscore
    text = re.sub(r'\d*', '', text)  # remove all numbers and words that contain numbers
    text = re.sub(' +', ' ', text)  # remove all additional spaces
    text = text.lower()
    return text


@jit
def stem(text):
    # print("Stemming...")
    stemmer = Stemmer()
    stemmed = ""
    for word in text.split():
        if word == 'docid':
            stemmed = stemmed + '\n'
        stemmed = stemmed + ' ' + stemmer.stem(word)

    return stemmed


def fix_topics(xml):
    result = ''
    m = None
    for line in xml.splitlines():
        if line.startswith('<'):
            if not(m is None) and m.group(1) != 'top':
                line = ' </' + m.group(1) + '>\n' + line
            m = re.search("\<([A-Za-z0-9_]+)\>", line)
            if not(m is None) and m.group(1) == 'top':
                line = '\n' + line + '\n'
        result = result + ' ' + line

    return result


@jit
def parse_query(root, option):

    result = ""
    queries = dict()
    for topic in root.find_all('top'):
        topic_id = topic.find('num').text.strip()
        topic_id = re.sub("[^0-9]", '', topic_id)  # keeps topic number only
        title = topic.find('title').text.strip()
        desc = topic.find('desc').text.strip()
        desc = desc.replace('Description', '')
        text = ""
        if option == "full":
            text = title + desc
        elif option == "title":
            text = title
        elif option == "description":
            text = desc
        text = clear_text(text)
        text = '\n' + topic_id + ' ' + text
        title = clear_text(title)
        desc = clear_text(desc)
        result = result + text
        # print("query id:", topic_id, "title", title)
        query = Query(topic_id, title, desc)
        queries.update({topic_id: query})

    return result, queries


@jit
def parse_docs(root):

    result = ""
    corpus_document = Corpus()

    for doc in root.find_all('DOC'):
        document_id = doc.find('DOCNO').text.strip()
        headline = ''
        content = ''
        if not (doc.find('HEADLINE') is None):
            headline = doc.find('HEADLINE').text.strip()
        if not (doc.find('TEXT') is None):
            content = doc.find('TEXT').text.strip()
        content = re.sub('<[^>]*>', ' ', content)
        text = headline + ' ' + content
        old_text = text
        text = clear_text(text)
        text = '\ndoc_id ' + document_id + ' ' + text
        headline = clear_text(headline)
        content = clear_text(content)
        # print("doc id:", document_id, "original text [:150]", old_text[:150], "text [:150]", text[:150])
        result = result + text
        if len(content) > 0:
            doc = Document(document_id, headline, content)
            # print("Headline:", headline, "Content:", content)
            corpus_document.add_doc(doc)

    return result, corpus_document


def load_from_pickle_file(filename):
    """
    :param filename: name of the pickle file
    :return: object to load
    """
    file = open(filename, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


def save_to_pickle_file(filename, obj):
    """
    :param filename: name of the pickle file
    :param obj: object to save
    :return: None
    """
    outfile = open(filename, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


def load_ids(retrieval_alg, n_pos, n_neg):
    """
    :return: List of 5 lists each (one per each fold) - training, validation and testing ids
    """
    labels_train_filename = "preprocessing/encoded_data/ids/cleared_ids_train_" + retrieval_alg + "_" + str(n_pos) + "_" + str(n_neg)
    labels_test_filename = "preprocessing/encoded_data/ids/cleared_ids_test_" + retrieval_alg
    ids_train = load_from_pickle_file(labels_train_filename)
    ids_test = load_from_pickle_file(labels_test_filename)
    return ids_train, ids_test


def score_to_text_run(scores, ids, config):
    """
    :param scores: scores in descending order for each topic
    :param ids: ids ordered according to scores
    :param config: configuration of DRMM run
    :return: text run
    """
    zipped = list(zip(ids, scores))
    filtered = []
    for _, group in groupby(sorted(zipped, key=lambda x: x[0]), key=lambda x: x[0]):
        g = list(group)
        # print(g)
        filtered.append(g[-1])
    '''for id_t in set([m[0] for m in zipped]):  # take set of ids in case of repetition
        lst = [m for m in zipped if m[0] == id_t]
        if len(lst) > 0:
            filtered.append(lst[0])'''
    text = ""
    for key, group in groupby(sorted(filtered, key=lambda x: x[0][0]), key=lambda x: x[0][0]):  # sort and group by topic
        i = 0
        for pair in sorted(group, key=lambda x: x[1], reverse=True)[:1000]:  # sort score desc
            topic_id, document_id = pair[0][0], pair[0][1]
            text += topic_id + " Q0 " + document_id + " " + str(i) + " " + str(pair[1]) + " DRMM_RUN_" + config + "\n"
            i += 1
    return text


def get_metrics_run(filename, qrels_path, test):
    """
    :param filename: name of run file
    :param qrels_path: path to qrels
    :return: map, p@20, nDCG@20 of the given run
    """
    result = subprocess.run("trec_eval.9.0.4/trec_eval -m map -m P -m ndcg_cut -m iprec_at_recall " + qrels_path + " " + filename,
                            shell=True, stdout=subprocess.PIPE)
    res = []
    prec_rec = []
    for line in result.stdout.decode('utf-8').split('\n'):
        if line.startswith("map") or line.startswith("P_20 ") or line.startswith("ndcg_cut_20 "):
            res.append(float(line.split()[2]))
        if line.startswith("iprec_at_recall_"):
            prec_rec.append(float(line.split()[2]))
    map = res[0]
    p20 = res[1]
    ndcg20 = res[2]
    if test:
        return float("%.3f" % map), float("%.3f" % p20), float("%.3f" % ndcg20), prec_rec
    else:
        return float("%.3f" % map), float("%.3f" % p20), float("%.3f" % ndcg20)


def embeddings(text, algo, sz, wdw, mc, mode, neg, subs, model_name):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # train model
    model = Word2Vec(text, sg=algo, size=sz, window=wdw, min_count=mc, hs=mode, negative=neg, sample=subs)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(model_name, "len:", len(words))
    # summarize the loaded model
    print(model)
    # save model
    model.save(model_name)
    return model


def load_models_w2v(queries_model_filename, corpus_model_filename):
    """
    :param queries_model_filename: file to open to retrieve queries w2v model
    :param corpus_model_filename: file to open to retrieve corpus w2v model
    :return: corpus and queries w2v models
    """
    query_model = Word2Vec.load(queries_model_filename)
    corpus_model = Word2Vec.load(corpus_model_filename)
    return query_model, corpus_model


def load_all_data(stopwords, stemmed, retrieval_alg):
    conf = ""
    if stopwords:
        conf = conf + "_stopwords"
    if stemmed:
        conf = conf + "_stemmed"

    corpus_filename = "preprocessing/pre_data/Corpus/Corpus" + conf
    queries_filename = "preprocessing/pre_data/Queries/Queries" + conf
    qrels_filename = "preranked/preranked_total_" + retrieval_alg
    corpus_model_filename = "preprocessing/pre_data/models/corpus_model" + conf + ".bin"
    queries_model_filename = "preprocessing/pre_data/models/queries_model" + conf + ".bin"

    corpus_obj = load_from_pickle_file(corpus_filename)
    queries_obj = load_from_pickle_file(queries_filename)
    qrels_obj = load_from_pickle_file(qrels_filename)
    queries_model, corpus_model = load_models_w2v(queries_model_filename, corpus_model_filename)
    return corpus_obj, queries_obj, qrels_obj, queries_model, corpus_model


def make_metric_plot(conf, all_losses_train, all_map_train, all_p20_train, all_ndcg20_train, all_map_val, all_p20_val, all_ndcg20_val, k):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_title("Training loss")
    ax1.plot(all_losses_train, 'r.-')
    ax2.set_title("Training map")
    ax2.plot(all_map_train, 'k.-')
    ax3.set_title("Training prec@20")
    ax3.plot(all_p20_train, 'b.-')
    ax4.set_title("Training nDCG@20")
    ax4.plot(all_ndcg20_train, 'g.-')
    plt.tight_layout()
    plt.savefig("plot_metrics/train_fold_" + str(k) + conf, bbox_inches="tight")
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title("Validation map")
    ax1.plot(all_map_val, 'k.--')
    ax2.set_title("Validation prec@20")
    ax2.plot(all_p20_val, 'b.--')
    ax3.set_title("Validation ndcg@20")
    ax3.plot(all_ndcg20_val, 'g.--')
    plt.tight_layout()
    plt.savefig("plot_metrics/val_fold_" + str(k) + conf, bbox_inches="tight")
    plt.clf()


def make_prec_recall_11pt_curve(conf, prec_rec, k):
    recall = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.plot(recall, prec_rec, '.-')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Interpolated recall-precision plot - fold " + str(k))
    plt.savefig("plot_metrics/recall_precision_fold_" + str(k) + conf)
    plt.clf()


def make_all_prec_recall_fold_curves(conf, all_prec_rec):
    recall = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color = ['r.-', 'k.-', 'b.-', 'g.-', 'y.-']
    for i in range(5):
        plt.plot(recall, all_prec_rec[i], color[i])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Interpolated recall-precision plot of all folds")
    plt.legend(["fold 1", "fold 2", "fold 3", "fold 4", "fold 5"], loc=0)
    plt.savefig("plot_metrics/interpolated recall-precision plot_" + conf)
    plt.clf()


def load_glove_model(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def plot_histograms(key, histogram, hist_mode, label, conf):
    topic = key[0]
    document = key[1]
    m = np.array(histogram)
    plt.matshow(m)
    plt.xticks([])
    plt.colorbar()
    plt.title("Matching topic " + topic + " with a " + label + " document " + document)
    plt.xlabel("opposite <- orthogonal -> similar (buckets)")
    plt.ylabel("Query terms")
    path_hist = "histograms/pictures/"
    plt.savefig(path_hist + str(key) + label + '_' + str(hist_mode) + conf)
    plt.close()
