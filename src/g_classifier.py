from ee_graph import EE
from reader import Reader
import os
import pandas as pd
import numpy as np
from itertools import combinations

class G_Classifier():
    """
    The class of the initial classifier g. This class first embeds all entity-entity pairs and then models z from the
    KG. The classifier f is then derived from this classifier.

    Attributes:
        graph: The entity-entity sub-graph of the KG
        embeddings: The embeddings of all words in the KG
        e_combinations: The combinations of all pairs of entities
        ee_embeddings: The embeddings of all entity-entity pairs
    """

    def __init__(self, lst_news):

        self.graph = EE(lst_news)
        self.embeddings = None #call function to get embeddings
        self.e_combinations = __create_combinations(self.graph.nodes)
        self.ee_embeddings = __embed_pairs(self.e_combinations)

    def __embed_pairs(self, pairs):
        """
        Private class method that embeds an entity-entity pair according to the function h.

        Args:
            pairs (list): A list containing entity-entity pairs

        Returns:
            list: A list containing the embeddings of the entity-entity pairs
        """

        embeddings = [self.__h(pair) for pair in pairs]
        return embeddings


    def __create_combinations(self, nodes):
        """
        Private class method that takes all entities and creates every entity-entity pair

        Args:
            nodes (dict): A dictionary that contains all known entities and all new entities

        Returns:
            list: A list containing all combinations of every entity-entity pair
        """

        known = list(zip(*nodes['K']))[0]
        new = list(zip(*nodes['N']))[0]
        combined = known + new
        return combinations(combined, 2)

    def __h(self, x, y):
        """
        Private class method that combines two word embeddings according to function defined in the paper (average)

        Args:
            x (np array): The embedding of the first word
            y (np array): The embedding of the second word

        Returns:
            np array: The embedding of the entity-entity pair
        """

        return 0.5 * np.add(x, y)




if __name__ == "__main__":
    # root = os.getcwd()
    # reader = Reader()
    # reader.read_files(root + '/data/bbc/business/')
    g = G(["Trump demands trade war with China Hazzaaaa"])

