from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np
import json


def display_whole_plot(model):

    vocab = list(model.wv.vocab)
    X = model[vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    words = list(vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    nameFig = "histograms/pictures/WholeModel.png"
    plt.savefig(nameFig)
    plt.clf()


def display_closestwords_tsnescatterplot(model, word, emb_size):

    arr = np.empty((0, emb_size), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    nameFig = "histograms/pictures/" + word + "Similarities.png"
    plt.savefig(nameFig)


with open('config.json') as config_file:
    data = json.load(config_file)

conf = data["conf"]
emb_size = data["emb_size"]
model = Word2Vec.load("preprocessing/pre_data/models/corpus_model" + conf + ".bin")
word = input("Enter the word you wish to explore:\n")
display_closestwords_tsnescatterplot(model, word, emb_size)
# display_whole_plot(model)  # too big