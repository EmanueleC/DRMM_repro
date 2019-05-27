from drmm import DRMM
from utilities.utilities import load_from_pickle_file, load_ids, score_to_text_run, get_metrics_run, make_metric_plot,\
    make_prec_recall_11pt_curve, make_all_prec_recall_fold_curves
from histograms.matching_histograms import MatchingHistograms
import tensorflow as tf
import numpy as np
import time
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
num_epochs = data["num_epochs"]
conf = data["conf"]
glv =data["use_glove"]

padded_query_idfs_filename = "preprocessing/encoded_data/idfs/padded_query_idfs" + conf
padded_query_embs_filename = "preprocessing/encoded_data/embeddings/padded_query_embs" + conf
histograms_total_filename = "preprocessing/encoded_data/histograms/histograms_total" + conf + "_glove_" + str(glv) + "_" + histograms_mode
qrels_path = "preprocessing/pre_data/Qrels/Qrels_cleaned.txt"

padded_query_idfs = load_from_pickle_file(padded_query_idfs_filename)
padded_query_embs = load_from_pickle_file(padded_query_embs_filename)
histograms_total = load_from_pickle_file(histograms_total_filename)  # 1.2 gb!

max_query_len = len(list(padded_query_idfs.values())[0])

print("max query len:", max_query_len)

ids_train, _, ids_test = load_ids()

count = 0
for fold in ids_train:
    print("Len fold", len(fold))
    count += len(fold)

matching_histograms = MatchingHistograms(num_bins, max_query_len)

all_map_test = []
all_p20_test = []
all_ndcg20_test = []
all_prec_rec_test = []
for k in range(5):
    ids_train_fold = ids_train[k]  # do. not. shuffle. (see loss function)
    ids_test_fold = ids_test[k]
    with tf.Session() as session:
        tf.random.set_random_seed(SEED)

        model = DRMM(num_layers, units, activation_functions, max_query_len, num_bins, emb_size, gating_function, SEED,
                     learning_rate)
        session.run(tf.global_variables_initializer())
        steps = len(ids_train_fold) - batch_size
        print("Number of steps per epoch:", int(steps/batch_size))
        all_losses = []
        all_map_train = []
        all_p20_train = []
        all_ndcg20_train = []
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = 0

            i = 0
            sims_train_epoch = []
            while i < steps:
                start = i
                end = i + batch_size

                batch_hist = []
                batch_idf = []
                batch_emb = []
                for (query_id, document_id) in ids_train_fold[start:end]:
                    '''query = queries.get(query_id)
                    document = corpus.get(document_id)
                    oov_document = oov_corpus.get(document_id)
                    oov_query = oov_queries.get(query_id)'''
                    hist = histograms_total[(query_id, document_id)]
                    batch_hist.append(hist)
                    batch_idf.append(padded_query_idfs.get(query_id))
                    batch_emb.append(padded_query_embs.get(query_id))

                assert np.array(batch_hist).shape[1:] == model.matching_histograms.shape[1:]
                assert np.array(batch_idf).shape[1:] == model.queries_idf.shape[1:]
                assert np.array(batch_emb).shape[1:] == model.queries_embeddings.shape[1:]
                sims_batch_train, _, c = session.run([model.sims, model.optimizer, model.loss],
                                                     feed_dict={model.matching_histograms: batch_hist,
                                                                model.queries_idf: batch_idf,
                                                                model.queries_embeddings: batch_emb})
                assert len(sims_batch_train) == batch_size
                sims_train_epoch += list(sims_batch_train)
                epoch_loss += c
                i += batch_size

            print('Epoch %s, loss=%2.4f, time=%4.4fs' % (epoch, epoch_loss, time.time() - start_time))
            all_losses.append(epoch_loss)
            train_epoch_run_text = score_to_text_run(sims_train_epoch, ids_train_fold, "sw_st_idf_ch")
            with open("run_results/training/train_epoch_run_" + str(k) + ".txt", 'w') as file:
                file.write(train_epoch_run_text)
            map, p20, ndcg20 = get_metrics_run("run_results/training/train_epoch_run_" + str(k) + ".txt", qrels_path, False)
            print(map, p20, ndcg20)
            all_map_train.append(map)
            all_p20_train.append(p20)
            all_ndcg20_train.append(ndcg20)

        make_metric_plot(all_losses, all_map_train, all_p20_train, all_ndcg20_train, k)

        hist_test = []
        idf_test = []
        emb_test = []
        for (query_id, document_id) in ids_test_fold:
            '''query = queries.get(query_id)
            document = corpus.get(document_id)
            oov_document = oov_corpus.get(document_id)
            oov_query = oov_queries.get(query_id)'''
            hist = histograms_total[(query_id, document_id)]
            hist_test.append(hist)
            idf_test.append(padded_query_idfs.get(query_id))
            emb_test.append(padded_query_embs.get(query_id))
        start_time = time.time()
        print("=== TESTING ===")
        predictions = session.run([model.sims], feed_dict={model.matching_histograms: hist_test, model.queries_idf: idf_test, model.queries_embeddings: emb_test})
        assert len(predictions[0]) == len(ids_test_fold)
        test_run_text = score_to_text_run(predictions[0], ids_test_fold, "sw_st_idf_ch")
        with open("run_results/test/test_run_" + str(k) + ".txt", 'w') as file:
            file.write(test_run_text)
        print("Testing required: %4.4fs" % (time.time() - start_time))
        map_t, p20_t, ndcg20_t, prec_rec_test = get_metrics_run("run_results/test/test_run_" + str(k) + ".txt", qrels_path, True)
        all_prec_rec_test.append(prec_rec_test)
        print(map_t, p20_t, ndcg20_t)
        all_map_test.append(map_t)
        all_p20_test.append(p20_t)
        all_ndcg20_test.append(ndcg20_t)

        make_prec_recall_11pt_curve(prec_rec_test, k)

print("Average MAP in folds:", sum(all_map_test)/len(all_map_test))
print("Average prec@20 in folds:", sum(all_p20_test)/len(all_p20_test))
print("Average nDCG@20 in folds:", sum(all_ndcg20_test)/len(all_ndcg20_test))
make_all_prec_recall_fold_curves(all_prec_rec_test)
