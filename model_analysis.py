import tensorflow as tf
from utilities.utilities import load_from_pickle_file
from drmm import DRMM
import json

with open('config.json') as config_file:
    data = json.load(config_file)

SEED = data["seed"]
stopwords = data["stopwords"]
stemmed = data["stemmed"]
histograms_mode = data["hist_mode"]
SEED = 42
num_layers = 3
units = [30, 5, 1]
activation_functions = ["tanh"] * num_layers
num_bins = 30
batch_size = data["batch_size"]
emb_size = 300
learning_rate = 1e-2
gating_function = data["gating_function"]
conf = data["conf"]

padded_query_idfs_filename = "preprocessing/encoded_data/idfs/padded_query_idfs" + conf
padded_query_idfs = load_from_pickle_file(padded_query_idfs_filename)
max_query_len = len(list(padded_query_idfs.values())[0])

model = DRMM(num_layers, units, activation_functions, max_query_len, num_bins, emb_size, gating_function, SEED,
             learning_rate)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
