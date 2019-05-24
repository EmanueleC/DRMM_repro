#!/usr/bin/env bash
cd ../terrier/bin
#use this file for the topics
echo trec.topics=../../../data/trec.robust.2004.txt >> ../etc/terrier.properties
#use this file for query relevance assessments
echo trec.qrels=../../../data/qrels.robust2004.txt >> ../etc/terrier.properties
echo matching.retrieved_set_size=2000 >> ../etc/terrier.properties
echo trec.output.format.length=2000 >> ../etc/terrier.properties
./trec_terrier.sh -r -Dtrec.model=DirichletLM
