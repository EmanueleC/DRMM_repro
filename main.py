from drmm import DRMM
from utilities.utilities import load_from_pickle_file, load_ids, score_to_text_run, get_metrics_run, make_metric_plot,\
    make_prec_recall_11pt_curve, make_all_prec_recall_fold_curves
from histograms.matching_histograms import MatchingHistograms
import tensorflow as tf
import numpy as np
import time
import json
import math

with open('config.json') as config_file:
    data = json.load(config_file)

SEED = data["seed"]
stopwords = data["stopwords"]
stemmed = data["stemmed"]
histograms_mode = data["hist_mode"]
min_delta = data["min_delta"]
patience = data["patience"]

num_layers = 3
units = [30, 5, 1]
activation_functions = ["tanh"] * num_layers
num_bins = 30
batch_size = data["batch_size"]
emb_size = 300
learning_rate = 1e-3

gating_function = data["gating_function"]
num_epochs = data["num_epochs"]
conf = data["conf"]
glv =data["use_glove"]

padded_query_idfs_filename = "preprocessing/encoded_data/idfs/padded_query_idfs" + conf
padded_query_embs_filename = "preprocessing/encoded_data/embeddings/padded_query_embs" + conf
histograms_total_filename = "preprocessing/encoded_data/histograms/histograms_total" + conf + "_glove_" + str(glv)\
                            + "_" + histograms_mode
qrels_path = "preprocessing/pre_data/Qrels/Qrels_cleaned.txt"

padded_query_idfs = load_from_pickle_file(padded_query_idfs_filename)
padded_query_embs = load_from_pickle_file(padded_query_embs_filename)
histograms_total = load_from_pickle_file(histograms_total_filename)  # 1.2 gb!

max_query_len = len(list(padded_query_idfs.values())[0])

print("max query len:", max_query_len)

ids_train, ids_validation, ids_test = load_ids()

matching_histograms = MatchingHistograms(num_bins, max_query_len)

all_map_test = []
all_p20_test = []
all_ndcg20_test = []
all_prec_rec_test = []
for k in range(5):
    ids_train_fold = ids_train[k]  # do NOT shuffle (see loss function)
    ids_val_fold = ids_validation[k]
    ids_test_fold = ids_test[k]
    best_val_map = - math.inf
    count_patience = 0
    with tf.Session() as session:
        tf.random.set_random_seed(SEED)

        model = DRMM(num_layers, units, activation_functions, max_query_len, num_bins, emb_size, gating_function, SEED,
                     learning_rate)
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        steps = len(ids_train_fold) - batch_size
        print("Number of steps per epoch:", int(steps/batch_size))
        all_losses_train = []
        all_map_train = []
        all_p20_train = []
        all_ndcg20_train = []
        all_losses_val = []
        all_map_val = []
        all_p20_val = []
        all_ndcg20_val = []
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_train_loss = 0
            epoch_val_loss = 0

            i = 0
            sims_train_epoch = []
            sims_val_epoch = []
            while i < steps:
                start = i
                end = i + batch_size

                batch_hist = []
                batch_idf = []
                batch_emb = []
                batch_hist_val = []
                batch_idf_val = []
                batch_emb_val = []

                for (query_id, document_id) in ids_train_fold[start:end]:
                    '''query = queries.get(query_id)
                    document = corpus.get(document_id)
                    oov_document = oov_corpus.get(document_id)
                    oov_query = oov_queries.get(query_id)'''
                    hist = histograms_total[(query_id, document_id)]
                    batch_hist.append(hist)
                    batch_idf.append(padded_query_idfs.get(query_id))
                    batch_emb.append(padded_query_embs.get(query_id))

                for (query_id, document_id) in ids_val_fold[start:end]:
                    '''query = queries.get(query_id)
                    document = corpus.get(document_id)
                    oov_document = oov_corpus.get(document_id)
                    oov_query = oov_queries.get(query_id)'''
                    hist = histograms_total[(query_id, document_id)]
                    batch_hist_val.append(hist)
                    batch_idf_val.append(padded_query_idfs.get(query_id))
                    batch_emb_val.append(padded_query_embs.get(query_id))

                assert np.array(batch_hist).shape[1:] == model.matching_histograms.shape[1:]
                assert np.array(batch_idf).shape[1:] == model.queries_idf.shape[1:]
                assert np.array(batch_emb).shape[1:] == model.queries_embeddings.shape[1:]
                assert np.array(batch_hist_val).shape[1:] == model.matching_histograms.shape[1:]
                assert np.array(batch_idf_val).shape[1:] == model.queries_idf.shape[1:]
                assert np.array(batch_emb_val).shape[1:] == model.queries_embeddings.shape[1:]
                sims_batch_train, _, c_train = session.run([model.sims, model.optimizer, model.loss],
                                                           feed_dict={model.matching_histograms: batch_hist,
                                                                      model.queries_idf: batch_idf,
                                                                      model.queries_embeddings: batch_emb})
                sims_batch_val, c_val = session.run([model.sims, model.loss],
                                                    feed_dict={model.matching_histograms: batch_hist_val,
                                                               model.queries_idf: batch_idf_val,
                                                               model.queries_embeddings: batch_emb_val})
                assert len(sims_batch_train) == batch_size
                sims_train_epoch += list(sims_batch_train)
                sims_val_epoch += list(sims_batch_val)
                epoch_train_loss += c_train
                epoch_val_loss += c_val
                i += batch_size

            print('Epoch %s' % epoch)
            print('train_loss=%2.4f, time=%4.4fs' % (epoch_train_loss, time.time() - start_time))
            print('val_loss=%2.4f, time=%4.4fs' % (epoch_val_loss, time.time() - start_time))
            all_losses_train.append(epoch_train_loss)
            all_losses_val.append(epoch_val_loss)
            start_time = time.time()
            train_epoch_run_text = score_to_text_run(sims_train_epoch, ids_train_fold, "sw_st_idf_lch")
            val_epoch_run_text = score_to_text_run(sims_val_epoch, ids_val_fold, "sw_st_idf_lch")
            with open("run_results/training/train_epoch_run_" + str(k) + ".txt", 'w') as file:
                file.write(train_epoch_run_text)
            with open("run_results/validation/val_epoch_run_" + str(k) + ".txt", 'w') as file:
                file.write(val_epoch_run_text)
            map_train, p20_train, ndcg20_train = get_metrics_run("run_results/training/train_epoch_run_" + str(k) +
                                                                 ".txt", qrels_path, False)
            print('train map=%2.4f, p@20=%2.4f, ndcg@20=%2.4f, time=%4.4fs' % (map_train, p20_train, ndcg20_train, time.time() - start_time))
            map_val, p20_val, ndcg20_val = get_metrics_run("run_results/validation/val_epoch_run_" + str(k) +
                                                           ".txt", qrels_path, False)
            print('val map=%2.4f, p@20=%2.4f, ndcg@20=%2.4f, time=%4.4fs' % (map_val, p20_val, ndcg20_val, time.time() - start_time))
            if map_val - best_val_map < min_delta:  # early stopping
                if count_patience < patience:
                    count_patience += 1
                else:
                    print("early stopping: restoring model with best map!")
                    saver.restore(session, "models/model.ckpt")
                    break
            if map_val > best_val_map:  # save model with best validation map
                best_val_map = map_val
                save_path = saver.save(session, "models/model.ckpt")
            all_map_train.append(map_train)
            all_p20_train.append(p20_train)
            all_ndcg20_train.append(ndcg20_train)
            all_map_val.append(map_val)
            all_p20_val.append(p20_val)
            all_ndcg20_val.append(ndcg20_val)

        make_metric_plot(all_losses_train, all_map_train, all_p20_train, all_ndcg20_train, all_losses_val, all_map_val,
                         all_p20_val, all_ndcg20_val, k)

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
