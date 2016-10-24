import csv
import math
from textblob import TextBlob as tb

# http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

f = open('gatech-aug2016.csv', 'r')
reader = csv.reader(f)
rownum = 0
header = ''
txt = ''
for row in reader:
    # Save header row.
    if rownum == 0:
        header = row
        break
    else:
        colnum = 0
        for col in row:
            if col == 9 or col == 10:
                txt += col
    rownum += 1
blobs = tb(txt)
for i, blob in enumerate(blobs):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, blobs) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

'''for i, v in enumerate(header): #9, 10
    print i, v
'''