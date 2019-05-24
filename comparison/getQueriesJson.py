from utilities.utilities import fix_topics, clear_text
from bs4 import BeautifulSoup
import re
import json
import pickle

queries_file = open('preprocessing/pre_data/Queries/Queries', 'rb')
queries_obj = pickle.load(queries_file)
queries_file.close()

jsonQueries = dict()
jsonQueries["queries"] = []

stopwords = True
title = True
description = False

topic_path = "data/trec.robust.2004.txt"

with open(topic_path, 'r') as f:
    xml = f.read()

xml = fix_topics(xml)

xml = "<?xml version='1.0' encoding='utf8'?>\n" + '<root>' + xml + '</root>'  # Add a root tag
root = BeautifulSoup(xml, "xml")

for topic in root.find_all('top'):
    query = dict()
    topic_id = topic.find('num').text.strip()
    topic_id = re.sub("[^0-9]", '', topic_id)  # keeps topic number only
    query["number"] = topic_id
    if title:
        text = topic.find('title').text.strip()
    elif description:
        text = topic.find('desc').text.strip()
    text = clear_text(text)
    if stopwords:
        text = "#stopword(" + text + ")"
    query["text"] = text
    jsonQueries["queries"].append(query)

outfile_name = "comparison/queries"
if title:
    outfile_name = outfile_name + "OnlyTitle"
elif description:
    outfile_name = outfile_name + "OnlyDescription"
else:
    outfile_name = outfile_name + "FullText"
if stopwords:
    outfile_name = outfile_name + "Stopwords"
outfile_name = outfile_name + ".json"

with open(outfile_name, 'w') as outfile:
    json.dump(jsonQueries, outfile, indent=4, ensure_ascii=False)
