# DRMM_repro
Implementation of a "Deep Relevance Matching Model" (DRMM) for Ad-hoc Retrieval
## References:
"A Deep Relevance Matching Model for Ad-hoc Retrieval" - Jiafeng Guo, Yixing Fan, Qingyao Ai, W. Bruce Croft (https://arxiv.org/abs/1711.08611)

### Readme (English)

The repository is organized as follows:

- data: here there will be TIPSTER (TREC Robust04) collection (the initial corpus)
- preprocessing/pre_data contains the following directories: Corpus, Queries, Qrels, idfs, ids, models and vocabulary. Its contents are textual data serialized in objects of custom classes)
- preranked contains 2000 top documents retrieved with anserini Bm25 and DiricheletLM
- preprocessing/encoded_data contains Corpus, Queries, idfs, ids, embeddings, histograms and vocabulary. Its contents are textual data encoded with respect to a dictionary of words
Bm25 (o QL)/run_results contains the runs produced during training and testing of the model, depending on the starting run.
- plot_metrics contains four plot with the loss, MAP, prec@20 and nDCG@20 obtained during the 5-fold cross validation and the precision-recall curves.
- utilities: various functions used for minor tasks
- data_analysis: contains the code for inspect the text collection.

The runs to re-rank are already in the repository. They have been evaluated with trec eval and gave the following results:


\ | Terrier DirichletLM(mu=2500) | Terrier Bm25(k_1 = 1.2, k_3 = 8, b = 0.75)
--------------------------|--------------------------|--------------------------
MAP| 0.241 | 0.247
Prec@20| 0.404 | 0.417
nDCG@20| 0.343 | 0.359

In order to run DRMM, the following steps must be executed in order:

```
python -m preprocessing.pre
python -m comparison.getRunFromFile
python -m preprocessing.stem_stopwords
python -m preprocessing.tf-idf
python -m preprocessing.prepare_ids
python -m preprocessing.embeddings
python -m preprocessing.encoding
python -m histograms.save_hist
python main.py # execute the model
```

To run dataset analysis:

```
python -m utilities.data_analysis
```

To explore different settings of DRMM, one may change the config.json file, which contains the following options: initial retrieval algorithm to use, seed, stopwords removal, stemming, embeddings size, histograms modalities, gating function, number of epochs, mini batch size, learning rate, and more.

A constraint to satisfy is the following: # doc pos + # doc neg must be a multiple of the mini batch size, so that, at each batch the loss is comput w.r.t. a given query.

### Readme (Italiano)

Implementazione di DRMM "Deep Relevance Matching Model" per task di Ad-Hoc Retrieval.

Il repository è organizzato come segue:

- data: qui va messa la collezione TIPSTER (TREC Robust04) (il corpus iniziale).
- preprocessing/pre_data: Corpus, Queries, Qrels, idfs, ids, models, vocabulary (deve contenere i dati testuali serializzati e "puliti", stemmati, con rimozione delle stopwords, gli idf dei termini delle queries, gli embeddings, un vocabolario delle parole e gli id per training, validation e test set - già suddivisi in fold)
- preranked: contiene i top 2000 documenti reperiti con Bm25 o QL di Terrier
- preprocessing/encoded_data: Corpus, Queries, idfs, ids, embeddings, vocabulary, histograms (deve contenere i dati testuali codificati in numeri rispetto a un dizionario - vocabulary -, gli embeddings o gli idf per le query e gli istogrammi (l'input della rete))
Bm25 (o QL)/run_results: training, validation=test (run prodotte durante l'allenamento e il testing), per ogni giro di cross validation
plot_metrics (mostra 4 grafici con la loss, MAP, prec@20 e nDCG@20 per la 5-fold cross validation, e le curve di precisione-richiamo per ciascuna run ottenuta dai fold di test)
- utilities: funzioni di utilità; analisi del dataset
- comparison: analizza le run da re-rankare, le serializza in oggetti di classi da me definiti
- data_analysis: deve contenere grafici per l'analisi del dataset

Le run da re-rankare sono già presenti nel repository. Le metriche ottenute sono le seguenti:

\ | Terrier DirichletLM(mu=2500) | Terrier Bm25(k_1 = 1.2, k_3 = 8, b = 0.75)
--------------------------|--------------------------|--------------------------
MAP| 0.241 | 0.247
Prec@20| 0.404 | 0.417
nDCG@20| 0.343 | 0.359

Per eseguire DRMM occorre dare i seguenti comandi nell'ordine in cui sono presentati:

```
python -m preprocessing.pre # per parsare e pulire documenti e queries e serializzarli in oggetti di classi da me definiti
python -m comparison.getRunFromFile  # per serializzare le run da re-rankare in oggetti di classi da me definiti
python -m preprocessing.stem_stopwords # per rimuovere stopwords dalle queries e stemmare queries e documenti
python -m preprocessing.tf-idf # calcola gli idf per i termini delle query
python -m preprocessing.prepare_ids # salva gli id degli esempi da utilizzare per l'allenamento di DRMM
python -m preprocessing.embeddings # in alternativa, caricare dei pre-trained embeddings da fonti esterne
python -m preprocessing.encoding  # codifica i dati precedenti in dizionari
python -m histograms.save_hist
python main.py # esegue il modello
```

Per eseguire l'analisi dei dati:

```
python -m utilities.data_analysis
```

Per provare diverse configurazioni di DRMM, è possibile cambiare il file config.json, che contiene le seguenti opzioni: tipo di algoritmo di reperimento da usare per il pre-rank, seed, rimozione stopwords, stemming, dimensioni embeddings, modalità istogrammi, funzione di gating, numero di epoche, dimensione dei mini batch, learning rate, e altre ancora.

Il vincolo da rispettare è che la somma del numero di documenti positivi e il numero di documenti negativi scelti sia un multiplo della dimensione dei mini batch, così un mini batch riguarda solo una query.
