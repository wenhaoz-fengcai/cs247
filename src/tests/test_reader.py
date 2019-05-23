import os
import sys
import logging
import unittest
from src.reader import Reader
from nltk.tokenize import word_tokenize

TEXT = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
TEST1 = [('While', 'O'), ('in', 'O'), ('France', 'LOCATION'), (',', 'O'), ('Christine', 'PERSON'), ('Lagarde', 'PERSON'), ('discussed', 'O'), ('short-term', 'O'), ('stimulus', 'O'), ('efforts', 'O'), ('in', 'O'), ('a', 'O'), ('recent', 'O'), ('interview', 'O'), ('with', 'O'), ('the', 'O'), ('Wall', 'ORGANIZATION'), ('Street', 'ORGANIZATION'), ('Journal', 'ORGANIZATION'), ('.', 'O')]

class TestReader(unittest.TestCase):
    def setUp(self):
        self.reader = Reader()
        self.tokenized_text = word_tokenize(TEXT)
        self.classified_text = self.reader.st.tag(self.tokenized_text)

    def test_init(self):
        assert TEST1 == self.classified_text

    def test_read_files(self):
        self.lst_news = self.reader.read_files("data/bbc")
        self.assertFalse(len(self.reader.file_names) == 0)
        self.assertTrue(os.access(self.reader.file_names[0], os.R_OK))

    def test_parse_news(self):
        self.lst_news = self.reader.read_files("data/bbc")
        # test on a subset of news articles, e.g. 10 files
        res = self.reader.parse_news(self.lst_news[:10])

    def test_filter_stop_words(self):
        example = ['This', 'is', 'a', 'sample', 'sentence', ',', 'showing',  'off', 'the', 'stop', 'words', 'filtration', '.']
        res = self.reader.filter_stop_words(example)
        print(res)
        assert res == ['sample', 'sentence', ',', 'showing', 'stop', 'words', 'filtration', '.']

    def test_stem_words(self):
        example = ['game', 'gaming', 'gamed', 'games']
        res = self.reader.stem_words(example)
        assert res == ['game']

def main():
    test_reader = TestReader()

if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger("test_reader").setLevel( logging.DEBUG )
    unittest.main()