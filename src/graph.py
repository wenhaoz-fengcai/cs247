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
    Base class which represents the heterogeneous textual graph (undirected graph). G = <V, E>. V is the set of nodes (objects), including 3 types of objects (i.e. new entites, known entities, and contextual words). Entities are words (with label: PERSON, LOCATION, and ORGANIZATION) whereas contextual words are the remaining uni-gram words. New entities are the entities not in DBpedia, and Know entities are the entities in the DBpedia. E is a set of edges (co-occurrences) of entity-entity, entity-word, and word-word corrences. Words within every 5-word sliding window in a news sentence are considered to be co-occuring with each other. The weights are represented by adjacency matrix using dataframe.

    Attributes:
        nodes: dictionary of nodes {"N (new entity)": [(word, label)], "K (Known entity)": [(word, label)], "C (Contextual word)": [(word, label)]} in the graph; Includes 3 types of objects (i.e. e new entites, known entities, and contextual words).
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
            Returns a dictionary contains 3 types of objects (i.e. new entites, known entities, and contextual words). E.g. {"N": [("Washington", "LOCATION")], "K":[("Trump", "PERSON"), ("Hua Wei", "ORGANIZATION")], "C": [("the", "O"), ("am", "O")]}
        """

        # parse news articles
        tagged_words = self.reader.parse_news(self.news)
        # seperate entities from contextual words
        entities, cwords = self.__entities_words(tagged_words)
        new_e, known_e = self.search.query(entities)

        ret = dict()
        ret["N"] = list(set(new_e))
        ret["K"] = list(set(known_e))
        ret["C"] = list(set(cwords))

        return dict(ret)

    def get_nodes(self):
        """
        Getter method which returns all nodes from self.nodes.
        """
        ret = set()
        for i in self.nodes["N"]:
            ret.add(i[0]) 
        for i in self.nodes["K"]:
            ret.add(i[0])
        for i in self.nodes["C"]:
            ret.add(i[0])
        return list(ret)

    def get_entities(self):
        """
        Getter method which returns a list of entities (i.e. word tagged with "PERSON", "LOCATION", "ORGANIZATION") from self.nodes.
        """
        ret = set()
        for i in self.nodes["N"]:
            ret.add(i[0]) 
        for i in self.nodes["K"]:
            ret.add(i[0])
        return list(ret)

    def get_words(self):
        """
        Getter method which returns a list of contextual words from self.nodes.
        """
        ret = set()
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
                print(set(itertools.combinations(t, 2)))
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
        if len(result) <= n:
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
        words = self.get_nodes()
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

    def __entities_words(self, tagged_words):
        """Private class method
        Seperate the entity words from the comtextual words.

        Args:
            tagged_words: list of strings; a list of tuples (word, label)

        Returns:
            entities: words tagged with "PERSON", "LOCATION", "ORGANIZATION".
            cwords: words tagged with "O" 
        """          
        entities = list()
        cwords = list()
        for word in tagged_words:
            if word[1] == "O":
                # contextual words
                cwords.append(word)
            else:
                entities.append(word)
        assert len(entities) + len(cwords) == len(tagged_words)
        return entities, cwords

    def update_weight(self, e, w):
        """
        Update the edge weight in the enternal weight matrix. 
        Args:
            e: tuple; a tuple contains two nodes, e.g. ("A", "B")
            w: int; The new weight associated with e
        """
        if e[0] not in set(self.get_nodes()):
            raise ValueError("Node {} is not in the graph".format(str(e[0])))
        if e[1] not in set(self.get_nodes()):
            raise ValueError("Node {} is not in the graph".format(str(e[1])))

        if e in self.edges and w <= 0:
            self.edge_weights.loc[e[0], e[1]] = w
            self.edge_weights.loc[e[1], e[0]] = w
            self.edges.remove(e)
            self.edges.remove((e[1], e[0]))
        elif e in self.edges and w > 0:
            self.edge_weights.loc[e[0], e[1]] = w
            self.edge_weights.loc[e[1], e[0]] = w
        else:
            self.edge_weights.loc[e[0], e[1]] = w
            self.edge_weights.loc[e[1], e[0]] = w
            self.edges.add(e)