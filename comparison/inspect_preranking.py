from utilities.utilities import load_from_pickle_file
from tqdm import tqdm
import math
import json
import matplotlib.pyplot as plt

with open('config.json') as config_file:
    data = json.load(config_file)

retrieval_alg = data["retrieval_alg"]

qrels_filename = "preranked/preranked_total_" + retrieval_alg

qrels = load_from_pickle_file(qrels_filename)

topics = list(range(301, 451)) + list(range(601, 701))

topic_freq_pos = []
topic_freq_neg = []
pos = []
neg = []

x = list(map(str, list(range(301, 451)) + list(range(601, 701))))

for topic in tqdm(x):
    topic_freq_pos.append(math.log(len(qrels.get_relevant_docs(topic).keys()) + 1))
    pos.append((len(qrels.get_relevant_docs(topic).keys())*100)/len(qrels.get_pairs_topic(topic)))
    topic_freq_neg.append(math.log(len(qrels.get_non_relevant_docs(topic).keys()) + 1))
    neg.append((len(qrels.get_non_relevant_docs(topic).keys())*100)/len(qrels.get_pairs_topic(topic)))

p1 = plt.stackplot(x, topic_freq_pos, topic_freq_neg, labels=["Positive documents", "Negative documents"])

plt.legend(loc="best")
plt.xticks(x[::20], rotation='90')
plt.title("# (Logs) Documents retrieved per topics with " + retrieval_alg + " algorithm")
plt.xlabel('Topics IDs')
plt.ylabel('# documents retrieved')
plt.savefig("comparison/queries_frequencies" + retrieval_alg + ".png")
print("avg pos:", sum(pos) / len(pos))
print("avg neg:", sum(neg) / len(neg))
print("min pos:", min(pos), "max pos:", max(pos))
print("min neg:", min(neg), "max neg:", max(neg))
plt.close()