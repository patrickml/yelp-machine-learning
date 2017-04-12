import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = word_tokenize(sentence)

print(tokens)
