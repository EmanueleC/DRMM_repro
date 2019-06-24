from drmm import DRMM
from utilities.utilities import load_from_pickle_file, load_ids, score_to_text_run, get_metrics_run, make_metric_plot,\
    make_prec_recall_11pt_curve, make_all_prec_recall_fold_curves
from histograms.matching_histograms import MatchingHistograms
import tensorflow as tf
import numpy as np
import time
import json
import math


def cross_validation(k_folds, num_epochs, batch_size, ids_train, ids_test, str_config, histograms_mode, opt):
    print(conf)
    histograms_total_filename = "preprocessing/encoded_data/histograms/histograms_total" + config + "_glove_" + str(glv) \
                                + "_" + histograms_mode
    histograms_total = load_from_pickle_file(histograms_total_filename)  # 1.2 gb!
    all_map_test = []
    all_p20_test = []
    all_ndcg20_test = []
    all_prec_rec_test = []
    for k in range(k_folds):
        ids_train_fold = ids_train[k]  # do NOT shuffle (see loss function)
        ids_test_fold = ids_test[k]
        print("len train fold:", len(ids_train_fold))
        print("len test fold:", len(ids_test_fold))
        best_val_map = - math.inf
        count_patience = 0
        with tf.Session() as session:
            tf.summary.FileWriter("./graphs/fold" + str(k), session.graph)
            tf.random.set_random_seed(SEED)

            model = DRMM(num_layers, units, activation_functions, max_query_len, num_bins, emb_size, gating_function,
                         SEED, learning_rate, opt)
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())
            train_steps = len(ids_train_fold) - batch_size
            all_losses_train = []
            all_map_train = []
            all_p20_train = []
            all_ndcg20_train = []
            all_map_val = []
            all_p20_val = []
            all_ndcg20_val = []
            for epoch in range(num_epochs):
                start_time = time.time()
                epoch_train_loss = 0

                i = 0
                sims_train_epoch = []
                while i < train_steps:
                    start = i
                    end = i + batch_size

                    batch_hist = []
                    batch_idf = []
                    batch_emb = []

                    for (query_id, document_id) in ids_train_fold[start:end]:
                        hist = histograms_total[(query_id, document_id)]
                        batch_hist.append(hist)
                        batch_idf.append(padded_query_idfs.get(query_id))
                        batch_emb.append(padded_query_embs.get(query_id))

                    assert np.array(batch_hist).shape[1:] == model.matching_histograms.shape[1:]
                    assert np.array(batch_idf).shape[1:] == model.queries_idf.shape[1:]
                    assert np.array(batch_emb).shape[1:] == model.queries_embeddings.shape[1:]
                    sims_batch_train, _, c_train = session.run([model.sims, model.optimizer, model.loss],
                                                               feed_dict={model.matching_histograms: batch_hist,
                                                                          model.queries_idf: batch_idf,
                                                                          model.queries_embeddings: batch_emb})
                    # print(sims_batch_train)
                    assert len(sims_batch_train) == batch_size
                    sims_train_epoch += list(sims_batch_train)
                    epoch_train_loss += c_train
                    i += batch_size

                hist_val = []
                idf_val = []
                emb_val = []
                for (query_id, document_id) in ids_test_fold:
                    hist = histograms_total[(query_id, document_id)]
                    hist_val.append(hist)
                    idf_val.append(padded_query_idfs.get(query_id))
                    emb_val.append(padded_query_embs.get(query_id))
                sims_val = session.run([model.sims], feed_dict={model.matching_histograms: hist_val,
                                                                model.queries_idf: idf_val,
                                                                model.queries_embeddings: emb_val})
                print('Epoch %s' % epoch)
                print('train_loss=%2.4f, time=%4.4fs' % (epoch_train_loss, time.time() - start_time))
                all_losses_train.append(epoch_train_loss)
                start_time = time.time()
                train_epoch_run_text = score_to_text_run(sims_train_epoch, ids_train_fold, "sw_st_idf_lch")
                val_epoch_run_text = score_to_text_run(sims_val[0], ids_test_fold, "sw_st_idf_lch")
                with open(retrieval_alg + "/training/train_epoch_run_" + str(k) + ".txt", 'w') as file:
                    file.write(train_epoch_run_text)
                with open(retrieval_alg + "/validation/val_epoch_run_" + str(k) + ".txt", 'w') as file:
                    file.write(val_epoch_run_text)
                map_train, p20_train, ndcg20_train = get_metrics_run(retrieval_alg + "/training/train_epoch_run_" + str(k) +
                                                                     ".txt", qrels_path, False)
                print('train map=%2.4f, p@20=%2.4f, ndcg@20=%2.4f, time=%4.4fs' % (
                map_train, p20_train, ndcg20_train, time.time() - start_time))
                map_val, p20_val, ndcg20_val = get_metrics_run(retrieval_alg + "/validation/val_epoch_run_" + str(k) +
                                                               ".txt", qrels_path, False)
                print('val map=%2.4f, p@20=%2.4f, ndcg@20=%2.4f, time=%4.4fs' % (
                map_val, p20_val, ndcg20_val, time.time() - start_time))
                if map_val - best_val_map < min_delta:  # early stopping
                    if count_patience < patience:
                        count_patience += 1
                    else:
                        print("stopping training: no improvements!")
                        break
                if map_val > best_val_map:  # save model with best validation map
                    best_val_map = map_val
                    count_patience = 0
                    saver.save(session, "models/model.ckpt")
                all_map_train.append(map_train)
                all_p20_train.append(p20_train)
                all_ndcg20_train.append(ndcg20_train)
                all_map_val.append(map_val)
                all_p20_val.append(p20_val)
                all_ndcg20_val.append(ndcg20_val)

                tf.summary.scalar('loss', all_losses_train)
                tf.summary.merge_all()

            make_metric_plot(str_config, all_losses_train, all_map_train, all_p20_train, all_ndcg20_train, all_map_val, all_p20_val,
                             all_ndcg20_val, k)

            hist_test = []
            idf_test = []
            emb_test = []
            for (query_id, document_id) in ids_test_fold:
                hist = histograms_total[(query_id, document_id)]
                hist_test.append(hist)
                idf_test.append(padded_query_idfs.get(query_id))
                emb_test.append(padded_query_embs.get(query_id))
            start_time = time.time()
            print("=== TESTING ===")
            saver.restore(session, "models/model.ckpt")
            predictions = session.run([model.sims], feed_dict={model.matching_histograms: hist_test,
                                                               model.queries_idf: idf_test,
                                                               model.queries_embeddings: emb_test})
            assert len(predictions[0]) == len(ids_test_fold)
            test_run_text = score_to_text_run(predictions[0], ids_test_fold, "sw_st_idf_ch")
            with open(retrieval_alg + "/test/test_run_" + str(k) + ".txt", 'w') as file:
                file.write(test_run_text)
            print("Testing required: %4.4fs" % (time.time() - start_time))
            map_t, p20_t, ndcg20_t, prec_rec_test = get_metrics_run(retrieval_alg + "/test/test_run_" + str(k) + ".txt",
                                                                    qrels_path, True)
            all_prec_rec_test.append(prec_rec_test)
            print(map_t, p20_t, ndcg20_t)
            all_map_test.append(map_t)
            all_p20_test.append(p20_t)
            all_ndcg20_test.append(ndcg20_t)

            make_prec_recall_11pt_curve(str_config, prec_rec_test, k)

    average_map = sum(all_map_test) / len(all_map_test)
    average_prec = sum(all_p20_test) / len(all_p20_test)
    average_ndcg = sum(all_ndcg20_test) / len(all_ndcg20_test)

    # print("Average MAP in folds:", average_map)
    # print("Average prec@20 in folds:", average_prec)
    # print("Average nDCG@20 in folds:", average_ndcg)
    make_all_prec_recall_fold_curves(str_config, all_prec_rec_test)
    return all_map_test, all_p20_test, all_ndcg20_test, average_map, average_prec, average_ndcg


with open('config.json') as config_file:
    data = json.load(config_file)

SEED = data["seed"]
stopwords = data["stopwords"]
stemmed = data["stemmed"]
# histograms_mode = data["hist_mode"]
min_delta = data["min_delta"]
patience = data["patience"]
retrieval_alg = data["retrieval_alg"]

num_layers = 3
units = [30, 5, 1]
activation_functions = ["tanh"] * num_layers
num_bins = 30
# batch_size = data["batch_size"]
emb_size = 300
learning_rate = data["learning_rate"]

gating_function = data["gating_function"]
# num_epochs = data["num_epochs"]
config = data["conf"]
glv = data["use_glove"]

padded_query_idfs_filename = "preprocessing/encoded_data/idfs/padded_query_idfs" + config
padded_query_embs_filename = "preprocessing/encoded_data/embeddings/padded_query_embs" + config
qrels_path = "preprocessing/pre_data/Qrels/Qrels_cleaned.txt"

padded_query_idfs = load_from_pickle_file(padded_query_idfs_filename)
padded_query_embs = load_from_pickle_file(padded_query_embs_filename)

max_query_len = len(list(padded_query_idfs.values())[0])

print("max query len:", max_query_len)

# ids_train, ids_validation, ids_test = load_ids()

matching_histograms = MatchingHistograms(num_bins, max_query_len)

f = open("parameter-tuning.txt", "a+")

for conf in [(15, 20, (100, 100), "Adagrad", "lch")]:
    sample = conf[2]
    num_epoch = conf[0]
    batch_size = conf[1]
    opt = conf[3]
    histograms_mode = conf[4]
    str_config = "retr_" + retrieval_alg + "_sample_" + str(sample) + "_epoch_" + str(num_epoch) + "_bs_" + str(batch_size) + "histmode_" + str(histograms_mode) + "_term_" + gating_function + "_opt_" + opt
    ids_train, ids_test = load_ids(retrieval_alg, sample[0], sample[1])
    all_map_test, all_p20_test, all_ndcg20_test, average_map, average_prec, average_ndcg = cross_validation(5, num_epoch, batch_size, ids_train, ids_test, str_config, histograms_mode, opt)
    text = str_config + "\nAVERAGE MAP IN FOLDS: "
    text += "\n all map test:" + " ".join(map(str, all_map_test))
    text += "\n all_p20_test:" + " ".join(map(str, all_p20_test))
    text += "\n all_ndcg20_test:" + " ".join(map(str, all_ndcg20_test))
    text += "\n all average_map:" + str(average_map)
    text += "\n average_prec:" + str(average_prec)
    text += "\n average_ndcg:" + str(average_ndcg)
    print(text)
    f.write(text)
f.close()
