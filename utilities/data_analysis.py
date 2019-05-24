from utilities.utilities import load_all_data
from statistics import mean, stdev, median, mode, variance, StatisticsError
import argparse
import matplotlib.pyplot as plt


def print_info_length(data, name_list, attr, plot):
    stats = "\nLength of data in " + name_list + ": " + str(len(data))
    if plot:
        plt.clf()
        plt.plot(range(len(data)), data, linestyle='None', marker='x')
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
    except (StatisticsError, ZeroDivisionError):
        print("No statistics available")
    plt.savefig("data_analysis/" + name_list + "_" + attr)
    return stats


parser = argparse.ArgumentParser()

parser.add_argument('-sw', action="store_true", default=False, dest='sw')
parser.add_argument('-st', action="store_true", default=False, dest='st')

results = parser.parse_args()

stopwords = results.sw
stemmed = results.st

corpus_obj, queries_obj, qrels_obj, queries_model, corpus_model = load_all_data(stopwords, stemmed)

text = ""
text += "Total documents: " + str(len(corpus_obj.docs.keys()))
text += "\nTotal queries: " + str(len(queries_obj.keys()))

lines_corpus_splitted = [len(line) for line in corpus_obj]
lines_queries_splitted = [len(line.split()) for query in queries_obj.values() for line in query.get_lines()]

max_lines_queries = max([len(query.get_lines()) for query in queries_obj.values()])
max_lines_docs = max([len(doc.get_lines()) for doc in corpus_obj.docs.values()])

text += '\n' + str(sum(lines_corpus_splitted)) + " total words in corpus"
text += '\n' + str(sum(lines_queries_splitted)) + " total words in queries"

text += "\nShortest document text: " + str(min(lines_corpus_splitted)) + " words"
text += "\nLongest document text: " + str(max(lines_corpus_splitted)) + " words"
text += "\nShortest query text: " + str(min(lines_queries_splitted)) + " words"
text += "\nLongest query text: " + str(max(lines_queries_splitted)) + " words"

attr = ""

if stopwords:
    attr = attr + "_stopwords"
if stemmed:
    attr = attr + "_stemmed"

text += print_info_length(lines_corpus_splitted, "corpus docs" + attr, "words", True)
text += print_info_length(lines_queries_splitted, "queries" + attr, "words", True)

text += '\n' + str(corpus_model)

w1 = "night"

text += '\n' + str(corpus_model.wv.most_similar(positive=w1, topn=6))

with open("data_analysis/data_analysis" + attr + ".txt", 'w') as file:
    file.write(text)
