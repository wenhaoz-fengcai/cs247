import os
import nltk
import os.path
import pandas as pd
import numpy as np

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

class Reader(object):
    """
    Python class which reads all the news documents from given path, and 
    tokenize the documents using "nltk Name Entity Tokenizer (NER)" package.
    This class should read file recursively.

    Attributes:
        file_names: a list of paths to all .txt files.
        files: a list of news articles.
        root: current path; project root path if we ran this module at 
            top-level.
        st: StandfordNERTagger with 3 class model for recognizing locations, 
            persons, and organizations.
    """

    def __init__(self):
        """Inits Reader"""
        self.file_names = []
        self.files = []
        # Download Punkt Sentence Tokenizer
        nltk.download('punkt')
        self.root = os.getcwd()
        # 3 class model for recognizing locations, persons, and organizations
        model_path = os.path.join(self.root, "stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz")
        tagger_path = os.path.join(self.root, "stanford-ner/stanford-ner.jar")

        self.st = StanfordNERTagger(model_path, tagger_path, encoding='utf-8')

    def read_files(self, inpath):
        """
        Read paths of all ".txt" files to self.file_names, and all news article
        to self.files.

        Args:
            inpath: string; path at which to read news data.

        Returns:
            A copy of list of news articles read from 'inpath'
        """
        for dirName, subdirList, fileList in os.walk(inpath):
            for file in fileList:
                if ".txt" in file:
                    self.file_names.append(os.path.join(self.root, dirName, file))

        for filename in self.file_names:
            with open(filename, "r", encoding="latin1") as f:
                self.files.append(f.read())

        return list(self.files)

    def read_csv_file(self, filepath):
        '''
        (Mixed-news data are stored in csv format. For convinience, we only remain title information and they are already informative.)

        Read path of the csv news file(usually read only one file).

        Args:
            filepath: string; path at where to read mixed-news data.

        Returns:
            A copy of list of news title
        ''' 
        article = pd.read_csv(filepath)
        
        self.files = article.iloc[:,1].tolist()

        return self.files 

    def parse_news(self, lst_news):
        """
        Join all the news in lst_news as one article. Parse it using
        StanfordNERTagger for recognizing locations, persons, and organizations.

        Args:
            lst_news: list of strings; list of news articles.

        Returns:
            A copy of list of (key, value) pairs. Each token is tagged (using our 3 
            class model) with either 'PERSON', 'LOCATION', 'ORGANIZATION', 
            or 'O'. The 'O' simply stands for other, i.e., non-named entities.
        """
        sep = "\n"
        self.tokenized_text = word_tokenize(sep.join(lst_news))
        self.classified_text = self.st.tag(self.tokenized_text)

        return list(self.classified_text)