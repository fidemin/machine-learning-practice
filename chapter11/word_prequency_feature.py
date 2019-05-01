
import numpy as np

from data_preprocessing import vocabulary

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

print(features[:10])
print(labels[:10])
