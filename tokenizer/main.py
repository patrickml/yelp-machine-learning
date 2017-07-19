import nltk
import json
import os.path
import collections
import itertools
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Streams a json input file for a specific number of lines
# and returns the number of lines as json documents
def readDocs(fileName, limit = -1):
    docs = []
    with open(fileName, 'r') as f:
        for i, line in enumerate(f):
            if limit != -1 and i > limit:
                break
            docs.append(json.loads(line))
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
            t = token.lower()
            # if the token exists already we will increment it by one
            if t in tokens:
              tokens[t] = tokens[t] + 1
            # else we will create it in our token dict and set it to 1
            else:
               tokens[t] = 1

    return sortSummarized(tokens)

# Returns if the rating is good or bad
def goodOrBad(rating):
    return 0 if rating < 3 else 1

# Adds tokenMapping to the documents
def mapTokensInReview(tokenMap, docs):
    # get the keys of the tokens this will be a list of words
    keys = list(tokenMap.keys())
    mappedTokens = []
    # itterate over each document
    for doc in docs:
        # create a new list on the
        doc["tokenMapping"] = []
        # create a new set of uniq tokens
        uniq = set(doc["tokens"])
        # for each uniq token
        for token in uniq:
            try:
                # get the tokens index
                index = keys.index(token)
                mapped = [index, goodOrBad(doc["stars"]), doc["stars"], doc["tokens"].count(token)]
                # append to the mapping of the tokens the scores and information
                # about each token that was used in the review
                doc["tokenMapping"].append(mapped)
                # also append the mapped tokens to a universal list from all docs
                mappedTokens.append(mapped)
            except ValueError:
                # move on the token isn't in the list
                continue
    return mappedTokens

INDEX = 0
GOOD_BAD = 1
STARS = 2
COUNT = 3

def start():
    sourceFile = '/yelp-data/yelp-reviews.json'
    tokenizedFile = '/yelp-data/yelp-reviews-tokenized.json'
    tokenSummaryFile = '/yelp-data/yelp-reviews-token-summary.json'
    tokenMapped = '/yelp-data/yelp-reviews-token-mapped.json'
    limitedSummaryFile = '/yelp-data/results.html'
    summaryExist = os.path.exists(tokenSummaryFile)
    tokenizedExists = os.path.exists(tokenizedFile)
    tokenizedMappedExists = os.path.exists(tokenMapped)

    documentLimit = 5000
    summaryLimit = 1000
    matrixSplit = 2500

    docs = []
    summary = collections.OrderedDict()
    mappedTokens = []

    if tokenizedMappedExists:
        mappedTokens = readDocs(tokenMapped)
        summary = sortSummarized(readDocs(tokenSummaryFile, 1)[0])
        docs = readDocs(tokenizedFile, documentLimit)
    else:
        # check if we have previously summarized our records
        if summaryExist:
            print("Reading Summarized Tokenized File")
            summary = sortSummarized(readDocs(tokenSummaryFile, 1)[0])
            docs = readDocs(tokenizedFile, documentLimit)
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

        # Create a mapped listed of tokens
        mappedTokens = mapTokensInReview(summary, docs)


    # make the matrix an np matrix so that we can grab specific columns
    tokensMatrix = np.array(mappedTokens)

    x0 = tokensMatrix[:, np.newaxis, INDEX]
    y0 = tokensMatrix[:, GOOD_BAD]

    x = x0[:-matrixSplit]
    y = y0[:-matrixSplit]

    x1 = x0[-matrixSplit:]
    y1 = y0[-matrixSplit:]

    # select only a certain amount of tokens from the summary
    # so we can see the top x tokens
    limitedSummary = itertools.islice(summary.items(), 0, summaryLimit)

    # Create linear regression object
    regr = sklearn.linear_model.LinearRegression()

    print(x)

    # Train the model using the training sets
    regr.fit(x, y)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((regr.predict(x1) - y1) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x1, y1))

    # Plot outputs
    plt.scatter(x1, y1, color='black')
    plt.plot(x1, regr.predict(x1), color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    print("Writing Results:")
    print("- Writing Limited Summarized To HTML:")
    writeSummaryLimitedToHTML(limitedSummaryFile, limitedSummary)

    if tokenizedMappedExists == False:
        print("- Writing Mapped Tokens:")
        writeJSONTOFile(tokenMapped, mappedTokens)
    if tokenizedExists == False:
        print("- Writing Tokenized Documents:")
        writeJSONTOFile(tokenizedFile, docs)
    if summaryExist == False:
        print("- Writing Summarized:")
        writeJSONTOFile(tokenSummaryFile, [summary])

# run our script
start()
