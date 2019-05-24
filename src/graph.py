import nltk
import itertools
import numpy as np
import pandas as pd
from itertools import islice
from src.reader import Reader
from src.search import Search
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Graph(object):
    """
    Python class which represents the heterogeneous textual graph (undirected graph). G = <V, E>. V is the set of nodes (objects), including two types of objects (i.e. new entites, and contextual words). New entities are words (with label: PERSON, LOCATION, and ORGANIZATION) cannot be matched by DBpedia whereas contextual words are the remaining uni-gram words. E is a set of edges (co-occurrences) of entity-entity, entity-word, and word-word corrences. Words within every 5-word sliding window in a news sentence are considered to be co-occuring with each other. 
    # TO-DO:
    The weights are represented by adjacency matrix using dataframe.

    Attributes:
        nodes: dictionary of nodes {"N (new entity)": [(word, label)], "C (contextual word)": [(word, label)]} in the graph; Including two types of objects (i.e. new entites, and contextual words).
        edges: set contains the tuples. e.g. ("A", "B") indicates a link between node "A" and node "B".
        weights: The weights are represented by adjacency matrix using dataframe.
        news: list of news articles (articles are string type).
    """
    def __init__(self, lst_news, window_size=5):
        """Inits Graph
        Args:
            lst_news: list of string. list of news articles.
        """
        self.window_size = window_size
        self.news = list(lst_news)
        self.reader = Reader()
        self.search = Search()

        self.nodes = self.__create_nodes()
        self.edges = self.__create_edges() 
        self.edge_weights = self.__create_weights()


    def __create_nodes(self):
        """Private class method
        Takes in a list of news articles (articles are string types):
        1) tokenize the articles
        2) remove stopwords
        3) label words with 3 labels (i.e. PERSON, ORGANIZATION, LOCATION)
        4) Match entities (i.e. person, org, loc) against DBpedia

        Returns:
            Returns a dictionary contains two types of objects (i.e. new entites, and contextual words). E.g. {"N": [("Washington", LOCATION)], "C":[("Trump", PERSON), ("Hua Wei", ORGANIZATION)]}
        """

        # parse news articles
        tagged_words = self.reader.parse_news(self.news)
        new, contextual = self.search.query(tagged_words)

        ret = dict()
        ret["N"] = list(set(new))
        ret["C"] = list(contextual)

        return dict(ret)

    def get_words(self):
        """
        Getter method which returns a list of words from self.nodes.
        """
        ret = set()
        for i in self.nodes["N"]:
            ret.add(i[0]) 
        for i in self.nodes["C"]:
            ret.add(i[0])
        return list(ret)


    def __create_edges(self, window_size=5):
        """Private class method
        Takes in a list of news articles, and extract the co-occurring links between nodes. Nodes within 5-word sliding window in a news sentence are considered to be co-occuring with each other. The frequncies of nodes co-appearing in news sentences as weights of these links.  

        Returns:
            Returns a set of links between nodes. 
        """
        e = set()
        for article in self.news:
            self.tokenized_text = word_tokenize(article)
            self.tokenized_text = self.reader.filter_stop_words(self.tokenized_text)
            generator = self.sliding_window(self.tokenized_text, self.window_size)

            for t in generator:
                e = e.union(set(itertools.combinations(t, 2)))
        return set(e)
    
    def get_edges(self):
        """
        Getter method which returns a set of edges from self.edges.
        """
        return set(self.edges)

        
    def sliding_window(self, seq, n=5):
        """
        Returns a sliding window (of width n) over data from the iterable
           s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

        Args:
            seq: list of words; This is one news article splitted into a list of words.
            n: int; size of the sliding window

        Returns:
            An iterator contains all the sliced window. See the test case in `tests/test_graph.py` for more details.
        """
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result   

    def __create_weights(self):
        """Private class method
        Create weights matrix using pandas dataframe. The value at ith row and jth row is the counts of links (undirected) between node i and node j.
        
        Returns:
            Return a copy of dataframe representing the weights matrix.
        """
        words = self.get_words()
        df = pd.DataFrame(index=words,columns=words).fillna(0)

        for article in self.news:
            self.tokenized_text = word_tokenize(article)
            self.tokenized_text = self.reader.filter_stop_words(self.tokenized_text)
            generator = self.sliding_window(self.tokenized_text, self.window_size)

            for t in generator:
                for tup in set(itertools.combinations(t, 2)):
                    if tup[0] != tup[1]:
                        df.loc[tup[0], tup[1]] += 1
                        df.loc[tup[1], tup[0]] += 1

        return df.copy()

    def get_weights(self):
        return self.edge_weights.copy()