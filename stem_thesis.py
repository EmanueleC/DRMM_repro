from krovetzstemmer import Stemmer
from utilities.utilities import clear_text

txt = ""
f = open("thesis.txt", "r")
for x in f:
    txt += x

stemmer = Stemmer()

txt = clear_text(txt)

stopwords_list = []

with open("inquery", 'r') as f:
    for line in f.readlines():
        stopwords_list.append(line.strip('\n'))

stopwords_list = set(stopwords_list)

txt2 = ""
for word in txt.split():
    if word not in stopwords_list:
        txt2 += " " + stemmer.stem(word)

with open('thesis_stemmed.txt', 'w') as file:
    file.write(txt2)
