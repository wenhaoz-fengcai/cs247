import pandas as pd
from src.graph import Graph

class EC(Graph):
    """
    A subclass of Graph class. It represents the heterogeneous textual graph (undirected graph). G = <V, E>. V is the set of nodes (objects), including 3 types of objects (i.e. new entites, known entities, and contextual words). Entities are words (with label: PERSON, LOCATION, and ORGANIZATION) whereas contextual words are the remaining uni-gram words. New entities are the entities not in DBpedia, and Know entities are the entities in the DBpedia. E is a set of edges (co-occurrences) of entity-entity, entity-word, and word-word corrences. Words within every 5-word sliding window in a news sentence are considered to be co-occuring with each other. The weights are represented by adjacency matrix using dataframe.

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
            windown_size: int. Size of the sliding window
        """
        # call the ___init__ of the parent class
        Graph.__init__(self, lst_news, window_size)
