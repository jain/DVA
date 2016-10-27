import csv
import math
from textblob import TextBlob as tb
import sys  

# http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def getBlob(fname):
    #f = open('gatech-aug2016.csv', 'r')
    f = open(fname, 'r')
    reader = csv.reader(f)
    rownum = 0
    header = ''
    txt = ''
    for row in reader:
        # Save header row.
        if rownum == 0:
            header = row
        else:
            for i, col in enumerate(row):
                if i == 9:
                    txt += col
        rownum += 1
    print txt
    blob = tb(txt)
    f.close()
    return blob

# http://stackoverflow.com/questions/21393758/unicodedecodeerror-ascii-codec-cant-decode-byte-0xe5-in-position-0-ordinal
reload(sys)  
sys.setdefaultencoding('utf8')

blobs = [getBlob('gatech-aug2016.csv'), getBlob('joker-aug2016.csv')]

for i, blob in enumerate(blobs):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, blobs) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

'''for i, v in enumerate(header): #9, 10
    print i, v
'''