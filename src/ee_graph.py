import pandas as pd
from src.graph import Graph

class EE(Graph):
    """
    A subclass of Graph class. It represents the heterogeneous textual graph (undirected graph). G = <V, E>. V is the set of nodes (objects), including 2 types of objects (i.e. new entities, known entities). Entities are words (with label: PERSON, LOCATION, and ORGANIZATION) whereas contextual words are the remaining uni-gram words. New entities are the entities not in DBpedia, and Known entities are the entities in the DBpedia. E is a set of edges (co-occurrences) of entity-entity. Words within every 5-word sliding window in a news sentence are considered to be co-occuring with each other. The weights are represented by adjacency matrix using dataframe.

    Attributes:
        nodes: dictionary of nodes {"K (Known entity)": [(word, label)], "N (New entity)": [(word, label)]} in the graph; Includes 2 types of objects (i.e. new entities, known entities).
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

        # excluding all contextual words, and new entities
        self.nodes = self.__clean_nodes(self.nodes)
        self.edges = self.__clean_edges(self.edges) 
        self.edge_weights = self.__clean_weights(self.edge_weights)

    def __clean_nodes(self, nodes):
        """Private class method
        Mask off {"C": [(word, label)]}
        Args:
            nodes: dictionary of nodes. 

        Return:
            Returns a dictionary of nodes with {"C": [(word, label)]} excluded.
        """
        ret = dict()
        ret["K"] = list(nodes["K"])
        ret["N"] = list(nodes["N"])
        ret["C"] = list()

        return ret

    def __clean_edges(self, edges):
        """Private class method
        Remove all edges whose both ends are not in self.nodes["K"] or self.nodes["N"]
        Args:
            edges: set of tuples

        Return:
            Returns a set of edges whose both ends are in self.nodes["K"] or self.nodes["N"]
        """
        ret = set()
        for item in edges:
            if (self.__in_nodes(self.nodes["K"] + self.nodes["N"], item[0]) 
                and self.__in_nodes(self.nodes["K"] + self.nodes["N"], item[1]) ):
                ret.add(item)
        return ret

    def __clean_weights(self, edge_weights):
        """Private class method
        Set the weights to 0 for the edges that are not in this graph
        Args:
            edge_weights: dataframe; Contains the weights
        Return:
            Returns a weight matrix that has all the weights associated in self.edges.
        """
        words = self.get_nodes()
        df = pd.DataFrame(index=words,columns=words).fillna(0)
        for item in self.edges:
            df.loc[item[0], item[1]] = edge_weights.loc[item[0], item[1]]
            df.loc[item[1], item[0]] = edge_weights.loc[item[1], item[0]]
        return df

    def __in_nodes(self, nodes, target):
        for item in nodes:
            if target == item[0]:
                return True

        return False
