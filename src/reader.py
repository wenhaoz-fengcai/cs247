import os
import os.path
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

class Reader(object):
    """
    Python class which reads all the news documents from given path, and 
    tokenize the documents using "nltk Name Entity Tokenizer (NER)" package.
    This class should read file recursively.
    """

    def __init__(self, inpath):
        """Inits Reader with reading path"""
        self.inpath = inpath
        # Download Punkt Sentence Tokenizer
        nltk.download('punkt')
        root = os.getcwd()
        # 3 class model for recognizing locations, persons, and organizations
        model_path = os.path.join(root, "src/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz")
        tagger_path = os.path.join(root, "src/stanford-ner/stanford-ner.jar")

        self.st = StanfordNERTagger(model_path, tagger_path, encoding='utf-8')
