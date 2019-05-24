from utilities.utilities import load_from_pickle_file
import matplotlib.pyplot as plt
import numpy as np
import json


def cosine_score(x, y):
    xx, yy, xy = 0.0, 0.0, 0.0
    for i in range(len(x)):
        xx += x[i] * x[i]
        yy += y[i] * y[i]
        xy += x[i] * y[i]
    return float("%.2f" % (xy / np.sqrt(xx * yy)))  # just to show numerical data on plot


with open('config.json') as config_file:
    data = json.load(config_file)

conf = data["conf"]

vocabulary_filename = "preprocessing/encoded_data/vocabulary/word_index" + conf
queries_filename = "preprocessing/encoded_data/Queries/Queries_encoded" + conf
corpus_filename = "preprocessing/encoded_data/Corpus/Corpus_encoded" + conf
corpus_model_filename = "preprocessing/encoded_data/embeddings/word_embeddings" + conf

vocabulary = load_from_pickle_file(vocabulary_filename)
ivd = {v: k for k, v in vocabulary.items()}
queries = load_from_pickle_file(queries_filename)
corpus = load_from_pickle_file(corpus_filename)
corpus_model = load_from_pickle_file(corpus_model_filename)

sample_query = queries['301'][:3]
sample_document = corpus['FBIS3-10082'][20:40]

trace = []
for query_term in sample_query:
    qtv = corpus_model[query_term]
    trace_qt = []
    for doc_term in sample_document:
        trace_qt.append(cosine_score(qtv, corpus_model[doc_term]))
    trace.append(trace_qt)

trace = np.array(trace)

fig, ax = plt.subplots()
im = ax.imshow(trace)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Cosine distance", rotation=-90, va="bottom")

# We want to show all ticks...
ax.set_yticks(np.arange(len(sample_query)))
ax.set_xticks(np.arange(len(sample_document)))

# ... and label them with the respective list entries
ax.set_yticklabels([ivd[term] for term in sample_query])
ax.set_xticklabels([ivd[term] for term in sample_document])

# Rotate the tick labels and set their alignment
plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

'''for i in range(len(sample_query)):
    for j in range(len(sample_document)):
        text = ax.text(j, i, trace[i, j], ha="center", va="center", color="w")'''

ax.set_title("Cosine similarity between a query and a document")
fig.tight_layout()
plt.savefig("histograms/pictures/cos_sim_sample")