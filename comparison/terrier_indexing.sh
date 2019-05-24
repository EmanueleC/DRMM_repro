#!/usr/bin/env bash
cd ../terrier/bin
./trec_setup.sh ../../../data/TIPSTER
echo termpipelines=Stopwords, KrovetzStemmer >> ../etc/terrier.properties
./trec_terrier.sh -i
