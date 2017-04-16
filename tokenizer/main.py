import nltk
import json
import os.path
import collections
import itertools

# Streams a json input file for a specific number of lines
# and returns the number of lines as json documents
def readDocs(fileName, limit):
    docs = [None] * (limit+1)
    with open(fileName, 'r') as f:
        for i, line in enumerate(f):
            if i > limit:
                break
            docs[i] = json.loads(line)
    return docs

# Reads and tokenizes a certain about of documents
def readAndTokenize(fileName, limit):
    # Steam the docs until we reach our limit
    docs = readDocs(fileName, limit)

    # Tokenize each document and store on a new key the tokenized version
    # of that document
    for doc in docs:
        doc["tokens"] = nltk.word_tokenize(doc["text"])

    return docs

# Sorts the summarized documents
def sortSummarized(summary):
    return collections.OrderedDict(sorted(summary.items(), key=lambda t: t[1], reverse = True))

# Writes json documents into a file
def writeJSONTOFile(fileName, docs):
    with open(fileName, 'w', buffering=20*(1024**2)) as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')

# Creates a new summary file in html
def writeSummaryLimitedToHTML(fileName, summary):
    with open(fileName, 'w', buffering=20*(1024**2)) as f:
        # move the the start of the file
        f.seek(0)
        f.write("<table>\n")
        f.write("  <thead>\n")
        f.write("   <tr>\n")
        f.write("     <th>Token</td>\n")
        f.write("     <th>Count</td>\n")
        f.write("   </tr>\n")
        f.write("  </thead>\n")
        f.write("  <tbody>\n")
        for key, value in summary:
            f.write("   <tr>\n")
            f.write("     <td>" + str(key) + "</td>\n")
            f.write("     <td>" + str(value) + "</td>\n")
            f.write("   </tr>\n")
        f.write("  </tbody>\n")
        f.write("</table>\n")
        # remove any extra lines incase there was already a file
        f.truncate()

# Summarizes a list of documents and their tokens into a count
# based on the number of times a token is used
def summarizeTokenUsage(docs):
    tokens = {}
    # itterate over each document
    for doc in docs:
        # grab the tokens
        for token in doc["tokens"]:
            # if the token exists already we will increment it by one
            if token in tokens:
              tokens[token] = tokens[token] + 1
            # else we will create it in our token dict and set it to 1
            else:
               tokens[token] = 1

    return sortSummarized(tokens)

def start():
    # Download the punkt tokenizer and load it into nltk
    nltk.download('punkt')
    sourceFile = '/yelp-data/yelp-reviews.json'
    tokenizedFile = '/yelp-data/yelp-reviews-tokenized.json'
    tokenSummaryFile = '/yelp-data/yelp-reviews-token-summary.json'
    limitedSummaryFile = '/yelp-data/results.html'
    summaryExist = os.path.exists(tokenSummaryFile)
    tokenizedExists = os.path.exists(tokenizedFile)

    documentLimit = 50000

    # ask the user how many of the top tokens they would like to see
    summaryLimit = int(input("How many of the top tokens would you like to see: "))


    docs = []
    summary = collections.OrderedDict()

    # check if we have previously summarized our records
    if summaryExist:
      print("Reading Summarized Tokenized File")
      summary = sortSummarized(readDocs(tokenSummaryFile, 1)[0])
    else:
      # check if we should read the raw data source
      if tokenizedExists == False:
          print("Reading Raw Data File")
          docs = readAndTokenize(sourceFile, documentLimit)
      # if not we should read from the already tokenized source file
      else:
          print("Reading Existing Tokenized File")
          docs = readDocs(tokenizedFile, documentLimit)

      # create a new summary of our tokens
      summary = summarizeTokenUsage(docs)

    # select only a certain amount of tokens from the summary
    # so we can see the top x tokens
    limitedSummary = itertools.islice(summary.items(), 0, summaryLimit)

    print("Writing Results:")
    print("- Writing Limited Summarized To HTML:")
    writeSummaryLimitedToHTML(limitedSummaryFile, limitedSummary)

    if tokenizedExists == False:
      print("- Writing Tokenized Documents:")
      writeJSONTOFile(tokenizedFile, docs)
    if summaryExist == False:
      print("- Writing Summarized:")
      writeJSONTOFile(tokenSummaryFile, [summary])



# run our script
start()
