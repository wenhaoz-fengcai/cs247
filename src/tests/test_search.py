import os
import sys
import logging
import unittest
from src.search import Search

class TestSearch(unittest.TestCase):
    def setUp(self):
        self.search = Search()

    def test_search(self):
	    mylist = [("dLondon", "Location"), ("Obama", "Person"), ("UNICEF", "Organization"), ("jhfvjhegrfvh", "Location"), ("Noor_Nakhaei", "Person"), ("Center_For_Smart_Health", "Organization")]
	    new, existing = self.search.query(mylist)

	    assert existing == [('Obama', 'Person'), ('UNICEF', 'Organization')]
	    assert new == [("dLondon", "Location"), ("jhfvjhegrfvh", "Location"), ("Noor_Nakhaei", "Person"), ("Center_For_Smart_Health", "Organization")]

def main():
	test_search = TestSearch()

if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger("test_reader").setLevel( logging.DEBUG )
    unittest.main()