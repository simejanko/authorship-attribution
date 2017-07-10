import numpy as np
from bs4 import BeautifulSoup,Tag
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.io import mmwrite
from scipy import sparse
import re
from sklearn.pipeline import make_union
from sklearn.preprocessing import FunctionTransformer
import itertools
    

class DocReader():
    def __init__(self):
        self.le = LabelEncoder()

    def __call__(self, file_path, ground_truth = None):
        bs = BeautifulSoup(open(file_path), 'lxml')
        gt = None
        root = bs.training
        if not ground_truth is None:
            gt = BeautifulSoup(open(ground_truth), 'lxml')
            root = bs.testing

        documents = []
        authors = []
        for text in root.findAll('text'):
            if gt is None:
                authors.append(text.author['id'])
            else:
                authors.append(gt.find('text', {'file' : text['file']}).author['id'])

            documents.append(text.get_text())

        if gt is None:
            authors = self.le.fit_transform(authors)
        else:
            authors = self.le.transform(authors)
        return documents, authors.reshape(-1,1)


class StemTokenizer:
    def __init__(self):
        self.ss = SnowballStemmer('english')
        #self.non_an = re.compile('[^a-zA-Z0-9 ]')

    def __call__(self, doc):
        #doc = re.sub(self.non_an, ' ', doc)
        return [self.ss.stem(t) for t in word_tokenize(doc)]


def group_by_author(docs, authors):
    docs_authors = sorted(zip(authors, docs), key=lambda x:x[0])
    grouped_docs = []
    for key, group in itertools.groupby(docs_authors, key=lambda x: x[0]):
        grouped_docs.append(' '.join([x[1] for x in list(group)]))
    return grouped_docs

#for adding mail length and average word length to features
#didn't end up being useful
def calc_custom_features(docs):
    print(docs.shape)
    tokenize = StemTokenizer()
    docs_len = []
    docs_wlen = []
    docs_avg_w_len = []
    for doc in docs[0,:]:
        docs_len.append(len(doc))
        tokenized = tokenize(doc)
        docs_wlen.append(len(tokenized))
        docs_avg_w_len.append((sum([len(t) for t in tokenized])/len(tokenized))
                              if len(tokenized) != 0 else 0)
    return np.column_stack((docs_len, docs_wlen, docs_avg_w_len))

read_docs = DocReader()
documents, authors = read_docs('data/train/LargeTrain.xml')
documents_valid, authors_valid = read_docs('data/train/LargeValid.xml','data/train/GroundTruthLargeValid.xml')
documents += documents_valid
authors = np.vstack((authors, authors_valid))
grouped = group_by_author(documents, authors)
documents_test, authors_test = read_docs('data/test/LargeTest.xml','data/test/GroundTruthLargeTest.xml')


print("Read the documents, preforming count vectorization...")
#Most frequent n-grams extraction
count_1_vectorizer = CountVectorizer(tokenizer=StemTokenizer(), ngram_range=(1,1), max_features=500, lowercase=False)
count_2_vectorizer = CountVectorizer(tokenizer=StemTokenizer(), ngram_range=(2,2), max_features=500, lowercase=False)
count_3_vectorizer = CountVectorizer(tokenizer=StemTokenizer(), ngram_range=(3,3), max_features=250, lowercase=False)

#Person-specific words extraction
tfidf_vectorizer = TfidfVectorizer(tokenizer=StemTokenizer(), ngram_range=(1,1), stop_words='english')
tfidf_vectorizer.fit_transform(grouped)
words = tfidf_vectorizer.get_feature_names()
selected_words = [words[i] for i in np.unique(np.argsort(tfidf_vectorizer.fit_transform(grouped).A)[:,-100:])]
count_selected_vectorizer = CountVectorizer(tokenizer=StemTokenizer(), stop_words='english', vocabulary=selected_words)

count_vectorizer = make_union(count_1_vectorizer, count_2_vectorizer,count_3_vectorizer, count_selected_vectorizer)
#custum_features = FunctionTransformer(calc_custom_features)
#count_vectorizer = TfidfVectorizer(tokenizer=StemTokenizer(), ngram_range=(1,1), max_features=300, use_idf=False, norm='l2')
count_vectorizer.fit(documents)

#getting rid of duplicate features - from person-specific and most frequent word extraction
feature_names = [w.split('__',1)[1] for w in count_vectorizer.get_feature_names()]
_, unique_feature_idxs = np.unique(feature_names, return_index=True)

X = count_vectorizer.transform(documents).A[:, unique_feature_idxs]
#X_valid = count_vectorizer.transform(documents_valid).A[:, unique_feature_idxs]
X_test = count_vectorizer.transform(documents_test).A[:, unique_feature_idxs]

np.save('data/train.npy', np.hstack((X, authors)))
#np.save('data/valid.npy', np.hstack((X_valid, authors_valid)))
np.save('data/test.npy', np.hstack((X_test, authors_test)))
