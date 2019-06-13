from utilities.utilities import load_all_data
from statistics import mean, stdev, median, mode, variance, StatisticsError
from gensim.models import KeyedVectors
import json
import matplotlib.pyplot as plt


def print_info_length(labels, data, name_list, attr, plot):
    stats = "\nLength of data in " + name_list + ": " + str(len(data))
    if plot:
        plt.clf()
        plt.plot(labels, data, linestyle='None', marker='x')
        plt.title(name_list + attr)
        plt.xlabel("Observations")
        plt.ylabel('# ' + attr)
    try:
        stats += "\nAverage " + attr + " in " + name_list + " " + str(mean(data))
        stats += "\nMin " + attr + " in " + name_list + " " + str(min(data))
        stats += "\nMax " + attr + " in " + name_list + " " + str(max(data))
        stats += "\nStandard deviation " + attr + " in " + name_list + " " + str(stdev(data))
        stats += "\nMedian " + attr + " in " + name_list + " " + str(median(data))
        stats += "\nVariance " + attr + " in " + name_list + " " + str(variance(data))
        stats += "\nMode " + attr + " in " + name_list + " " + str(mode(data))
        if plot:
            plt.axhline(median(data), color='g', linestyle='--', label='median: ' + str("{0:.2f}".format(median(data))))
            plt.axhline(mean(data), color='r', linestyle='--', label='mean: ' + str("{0:.2f}".format(mean(data))))
            plt.legend(loc='best')
            plt.xticks(labels[::20], rotation='90')
    except (StatisticsError, ZeroDivisionError):
        print("No statistics available")
    plt.savefig("data_analysis/" + name_list + "_" + attr)
    return stats


with open('config.json') as config_file:
    data = json.load(config_file)

stopwords = data["stopwords"]
stemmed = data["stemmed"]
glv = data["use_glove"]
conf = data["conf"]
retrieval_algorithm = data["retrieval_alg"]

corpus_obj, queries_obj, qrels_obj, queries_model, corpus_model = load_all_data(stopwords, stemmed, retrieval_algorithm)

text = ""
text += "Total documents: " + str(len(corpus_obj.docs.keys()))
text += "\nTotal queries: " + str(len(queries_obj.keys()))

lines_corpus_splitted = [len(set(line)) for line in corpus_obj]
corpus_labels = [doc.document_id for doc in corpus_obj.docs.values()]
corpus_labels.sort()
lines_corpus_splitted = [x for _, x in sorted(zip(corpus_labels, lines_corpus_splitted))]
lines_queries_splitted = [len(set(query.title.split())) for query in queries_obj.values()]
queries_labels = [query.qId for query in queries_obj.values()]
lines_queries_splitted = [x for _, x in sorted(zip(queries_labels, lines_queries_splitted))]

max_lines_queries = max([len(query.get_lines()) for query in queries_obj.values()])
max_lines_docs = max([len(doc.get_lines()) for doc in corpus_obj.docs.values()])

text += '\n' + str(sum(lines_corpus_splitted)) + " total words in corpus"
text += '\n' + str(sum(lines_queries_splitted)) + " total words in queries"

text += "\nShortest document text: " + str(min(lines_corpus_splitted)) + " words"
text += "\nLongest document text: " + str(max(lines_corpus_splitted)) + " words"
text += "\nShortest query text: " + str(min(lines_queries_splitted)) + " words"
text += "\nLongest query text: " + str(max(lines_queries_splitted)) + " words"

vocab = set()
for doc in corpus_obj.docs.values():
    vocab.update(doc.get_text().split())

text += "\nOriginal length vocabulary: " + str(len(vocab)) + " words"

print("text analysis")

# text += print_info_length(corpus_labels, lines_corpus_splitted, "corpus docs" + conf, "words", True)
text += print_info_length(queries_labels, lines_queries_splitted, "queries" + conf, "words", True)

text += '\n' + str(corpus_model)

print("done.")

w1 = "night"

outv = KeyedVectors(300)
outv.vocab = corpus_model.wv.vocab  # same
outv.index2word = corpus_model.wv.index2word  # same
outv.syn0 = corpus_model.syn1neg  # different

text += '\nIN EMBEDDINGS COMPARISON:\n' + str(corpus_model.wv.most_similar(positive=[corpus_model[w1]], topn=6))
print("IN-IN done.")
text += '\nOUT EMBEDDINGS COMPARISON:\n' + str(outv.most_similar(positive=[outv[w1]], topn=6))
print("OUT-OUT done.")
text += '\nIN-OUT EMBEDDINGS COMPARISON:\n' + str(corpus_model.wv.most_similar(positive=[outv[w1]], topn=6))
print("IN-OUT done.")
text += '\nOUT-IN EMBEDDINGS COMPARISON:\n' + str(outv.most_similar(positive=[corpus_model[w1]], topn=6))
print("OUT-IN done.")

with open("data_analysis/data_analysis" + conf + ".txt", 'w') as file:
    file.write(text)
