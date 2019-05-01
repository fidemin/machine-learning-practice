import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data_preprocessing import vocabulary


def raw_term_prequency():
    features = []
    labels = []

    with open('./data/SMSSpamCollection') as f:
        for line in f:
            feature = np.zeros(len(vocabulary))
            splits = line.split()
            label = splits[0]
            words = splits[1:]

            if label == 'spam':
                labels.append(1)
            else:
                labels.append(0)

            for word in words:
                word = word.lower()
                feature[vocabulary[word]] += 1

            feature = feature / sum(feature)
            features.append(feature)

    return features, labels


def term_prequency():
    spam_header = 'spam\t'
    no_spam_header = 'ham\t'
    documents = []
    labels = []

    with open('./data/SMSSpamCollection') as f:
        for i, line in enumerate(f):
            if line.startswith(spam_header):
                labels.append(1)
                documents.append(line[len(spam_header):])
            elif line.startswith(no_spam_header):
                labels.append(0)
                documents.append(line[len(no_spam_header):])
            else:
                raise ValueError("bad spam header at line %d")

    vectorizer = CountVectorizer()
    term_counts = vectorizer.fit_transform(documents)
    vocabulary = vectorizer.get_feature_names()
    tf_transfomer = TfidfTransformer(use_idf=False)
    tf_transfomer.fit(term_counts)
    features = tf_transfomer.transform(term_counts)
    
    return vocabulary, features, labels


if __name__ == "__main__":
    voca, features, labels = term_prequency()
    with open('processed.pickle', 'wb') as f:
        pickle.dump((voca, features, labels), f)
    

