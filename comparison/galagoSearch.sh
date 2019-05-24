#!/bin/bash

cd ../../lemur-galago-3.8
GALAGO=./core/target/appassembler/bin/galago
QUERIES_FOLDER=../PythonImpl/comparison
INDEX=../data/TIPSTER-INDEX
TIPSTER=../data/TIPSTER

# Indexing

$GALAGO build --inputPath=$TIPSTER --indexPath=$INDEX --tokenizer/fields+docno --tokenizer/fields+head --tokenizer/fields+text --fileType=trectext --stemmedPostings=true --stemmer+krovetz

# Retrieval

# $GALAGO batch-search --verbose=true --requested=2000 --index=$INDEX $QUERIES_FOLDER/queriesOnlyTitleStopwords.json --defaultTextPart=postings.krovetz --stopwordlist=inquery > $QUERIES_FOLDER/resultOnlyTitleStemmedStopwordsTipsterQL.eval
# $GALAGO batch-search --verbose=true --requested=2000 --index=$INDEX $QUERIES_FOLDER/queriesOnlyTitleStopwords.json --scorer=bm25 --defaultTextPart=postings.krovetz --stopwordlist=inquery > $QUERIES_FOLDER/resultOnlyTitleStemmedStopwordsTipsterBm25.eval
