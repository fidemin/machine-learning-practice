
vocabulary = {}

with open('./data/SMSSpamCollection') as f:
    for line in f:
        splits = line.split()
        #label = splits[0]
        words = splits[1:]

        for word in words:
            word = word.lower()
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)
