#-------------------------------------------------------------------------
# AUTHOR: John Huang
# FILENAME: indexing.py
# SPECIFICATION: Aims to replicate the tf-idf calculation
# FOR: CS 4250- Assignment #1
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#Importing some Python libraries
import csv
import math

documents = []

#Reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

#Conducting stopword removal for pronouns/conjunctions. Hint: use a set to define your stopwords.
#--> add your Python code here
stopWords = {"I", "She", "he", "They", "and", "her", "their", "the"}

#Conducting stemming. Hint: use a dictionary to map word variations to their stem.
#--> add your Python code here
stemming = {
    "cats": "cat",
    "dogs": "dog",
    "loves": "love",
    "love": "love",
    "dog": "dog",
    "cat": "cat"
}

#Identifying the index terms.
#--> add your Python code here
terms = set()
tokens = []

for doc in documents:
    
    words = doc.split()
    filtered_words = [stemming.get(word, word) for word in words if word not in stopWords]  

    tokens.append(filtered_words) # appends a list of words that we will traverse through
    terms.update(filtered_words) # adds unique words to the set

#Building the document-term matrix by using the tf-idf weights.
# Compute the document-term matrix with TF-IDF values\
def compute_tf(term, doc):
    return doc.count(term) / len(doc)

def compute_idf(term, docs):
    doc_count = sum([1 for doc in docs if term in doc])
    return math.log(len(docs) / doc_count, 10)

docTermMatrix = []

#doc in tokens is each sentence for each doc after processing
for doc in tokens:
    tfidf = []
    for term in terms:
        tf = compute_tf(term, doc)
        idf = compute_idf(term, tokens)
        tfidf.append(tf * idf) 
    docTermMatrix.append(tfidf)

#Printing the document-term matrix.
print("\t", end="") #spacing

for term in terms:
    print(f"{term}\t", end="")
print()

for i, row in enumerate(docTermMatrix):
    print(f"Doc {i+1}\t", end="")

    for value in row:
        print(f"{value:.3f}\t", end="")  #prints tfidf to 3 decimals

    print()
