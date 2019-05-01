
import pickle
from sklearn.linear_model import LogisticRegression

with open('processed.pickle','rb') as f:
    vocabulary, features, labels = pickle.load(f)


total_num = len(labels)
mid_idx = total_num // 2
train_features = features[:mid_idx,:]
train_labels = labels[:mid_idx]
test_features = features[mid_idx:, :]
test_labels = labels[mid_idx:] 

classifier = LogisticRegression()
classifier.fit(train_features, train_labels)
print('train accuracy: %4.4f' % classifier.score(train_features, train_labels))
print('test accuracy: %4.4f' % classifier.score(test_features, test_labels))

weights = classifier.coef_[0, :]
pairs = []

for idx, value in enumerate(weights):
    pairs.append((abs(value), vocabulary[idx]))

pairs.sort(key=lambda x:x[0], reverse=True)
for pair in pairs[:20]:
    print('score %4.4f word: %s' % pair)
