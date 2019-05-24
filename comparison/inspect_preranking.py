import matplotlib.pyplot as plt

retrieval_alg = "QL"

preranked_filename = "comparison/terrier_preranked/DirichletLM_6.res"

indexes = {}

""" create runs objects from galago batch-search output """
with open(preranked_filename, 'r') as results:
    runsList = (line.split() for line in results)

    for line in runsList:
        if line[0] in indexes.keys():
            indexes[line[0]] = indexes[line[0]] + 1
        else:
            indexes[line[0]] = 0

print(indexes)
freq = [x for x in indexes.values() if x < 1999]
topics = [key for key in indexes.keys() if indexes[key] < 1999]
plt.bar(topics, freq)
plt.xticks(rotation='90')
plt.title("# Documents retrieved per topics (freq < 2000) with " + retrieval_alg + " algorithm")
plt.xlabel('Topics IDs')
plt.ylabel('# documents retrieved')
plt.savefig("comparison/queries_frequencies" + retrieval_alg + ".png")
plt.close()
