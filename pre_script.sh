#!/bin/bash

python -m preprocessing.pre
#python -m comparison.getRunFromFile
python -m preprocessing.stem_stopwords -sw -st
python -m preprocessing.tf-idf -sw -st
python -m preprocessing.prepare_ids -topic=-1
python -m preprocessing.embeddings -sw -st
python -m preprocessing.encoding -sw -st
