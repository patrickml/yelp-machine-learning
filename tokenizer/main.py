import nltk
import json

# Download the punkt tokenizer and load it into nltk
nltk.download('punkt')

# Streams a json input file for a specific number of lines
# and returns the number of lines as json documents
def read_docs(filename, limit):
    docs = [None] * (limit+1)
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i > limit:
                break
            docs[i] = json.loads(line)
    return docs

docs = read_docs('/yelp-data/yelp-reviews.json', 10)

for doc in docs:
    tokens = nltk.word_tokenize(doc["text"])
    print(tokens)
