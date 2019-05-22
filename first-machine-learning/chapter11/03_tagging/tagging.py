
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize

MODEL_PATH = '/Users/yunhongmin/stanford-postagger-full/models/english-bidirectional-distsim.tagger'
JAR_PATH = '/Users/yunhongmin/stanford-postagger-full/stanford-postagger-3.9.2.jar'

pos_tagger = StanfordPOSTagger(MODEL_PATH, JAR_PATH)
text = 'If you unpack the tar file, you should have everything needed. This software provides a GUI demo, a command-line interface, and an API. Simple scripts are included to invoke the tagger. For more information on use, see the included README.txt.'

tokens = word_tokenize(text)
print(tokens)
print()
print(pos_tagger.tag(tokens))
