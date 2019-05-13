import sys
import logging
import unittest
from src.reader import Reader
from nltk.tokenize import word_tokenize

TEXT = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
TEST1 = [('While', 'O'), ('in', 'O'), ('France', 'LOCATION'), (',', 'O'), ('Christine', 'PERSON'), ('Lagarde', 'PERSON'), ('discussed', 'O'), ('short-term', 'O'), ('stimulus', 'O'), ('efforts', 'O'), ('in', 'O'), ('a', 'O'), ('recent', 'O'), ('interview', 'O'), ('with', 'O'), ('the', 'O'), ('Wall', 'ORGANIZATION'), ('Street', 'ORGANIZATION'), ('Journal', 'ORGANIZATION'), ('.', 'O')]

class TestReader(unittest.TestCase):
    def setUp(self):
        self.reader = Reader("../../data/bbc/")
        self.tokenized_text = word_tokenize(TEXT)
        self.classified_text = self.reader.st.tag(self.tokenized_text)

    def test_init(self):
        assert TEST1 == self.classified_text



if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger("test_reader").setLevel( logging.DEBUG )
    unittest.main()