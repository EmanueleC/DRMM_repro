import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-sw', action="store_true", default=False, dest='sw')  # apply stopwords removal
parser.add_argument('-st', action="store_true", default=False, dest='st')  # apply stemming
parser.add_argument('-hist', choices=['ch', 'nh', 'lch'], dest='hist')  # histograms mode
parser.add_argument('-idf', action="store_true", default=False, dest='idf')  # use idf

results = parser.parse_args()

stopwords = results.sw
stemmed = results.st
hist_mode = results.hist
idf = results.idf

result_filename = "drmm_net/drmm_net_data/results/TREC_res"

if stopwords:
    result_filename = result_filename + "_stopwords"
if stemmed:
    result_filename = result_filename + "_stemmed"
result_filename = result_filename + "_ep10_bs20_" + hist_mode + "_-1"
if idf:
    result_filename = result_filename + "_idfTrue"
else:
    result_filename = result_filename + "_idfFalse"

for i in range(5):
    print("=== RESULTS fold" + str(i) + "===")
    result_filename_i = result_filename + "_fold_" + str(i) + "_results.eval"
    print(result_filename_i)
    with open(result_filename_i) as f:
        for line in f:
            if line.startswith("map") or line.startswith("P_20 ") or line.startswith("ndcg_cut_20 "):
                print(line.strip('\n'))
