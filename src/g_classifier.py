from ee_graph import EE
from kg_graph import KG
from reader import Reader
import os
import pandas as pd
import numpy as np
from itertools import combinations
from random import seed, shuffle
from time import time
from sklearn.neural_network import MLPClassifier

class G_Classifier():
    """
    The class of the initial classifier g. This class first embeds all entity-entity pairs and then models z from the
    KG. The classifier f is then derived from this classifier.

    Attributes:
        ee_graph: The entity-entity sub-graph of the KG
        embeddings: The embeddings of all words in the KG
        e_combinations: The combinations of all pairs of entities
        ee_embeddings: The embeddings of all entity-entity pairs
    """

    def __init__(self, lst_news):

        seed(time())
        self.ee_graph = EE(lst_news)
        self.classifier = MLPClassifier(hidden_layer_sizes=(10,10,10), solver)

        self.embeddings = None #call function to get embeddings
        self.known_comb, self.all_combinations = __create_combinations(self.ee_graph.nodes)
        self.ee_embeddings = __embed_pairs(self.all_combinations)
        self.S_known = self.__random_sample(self.ee_embeddings, self.ee_embeddings.shape[0] * 0.25)
        __train_classifier()

        self.eps = self.__generate_eps(self.sample)

    def __embed_pairs(self, pairs):
        """
        Private class method that embeds an entity-entity pair according to the function h.

        Args:
            pairs (list): A list containing entity-entity pairs

        Returns:
            Dataframe: A df containing the embeddings of the entity-entity pairs as values and indexed by e-e pair
        """

        embeddings = {pair, self.__h(pair) for pair in pairs}
        embeddings = pd.from_dict(embeddings, orient='index')
        return embeddings


    def __create_combinations(self, nodes):
        """
        Private class method that takes all entities and creates every entity-entity pair

        Args:
            nodes (dict): A dictionary that contains all known entities and all new entities

        Returns:
            list: A list containing all combiantions of known entity-entity pair
            list: A list containing all combinations of every entity-entity pair
        """

        known = list(zip(*nodes['K']))[0]
        new = list(zip(*nodes['N']))[0]
        combined = known + new
        return combinations(known, 2), combinations(combined, 2)

    def __h(self, x, y):
        """
        Private class method that combines two word embeddings according to function defined in the paper (average)

        Args:
            x (np array): The embedding of the first word
            y (np array): The embedding of the second word

        Returns:
            np array: The embedding of the entity-entity pair
        """

        return 0.5 * np.add(x, y) # forumula from paper

    def __random_sample(self, pairs, sample_size):
        """
        Obtain random sample of all entity pairs which are in the KG

        Args:
            pairs (Dataframe): The dataframe containing the embeddings of all entity-entity pairs
            sample_size (int): The size of the sample to take from the dataframe

        Returns:
            Dataframe: The subset of entity-entity pairs from the random sample that are in the KG
        """

        size_known = 0
        while size_known < 50:  # want a random sample of minimum size (min size is arbitrary)
            S = nodes.sample(sample_size)
            S_known = S.loc[self.known, :]
            size_known = S_known.shape[0]

        return S_known

    def __train_classifier(self):
        pass



    def __generate_eps(self, random_sample):
        """
        Private class method for calculating epsilon which is used to scale the classifier f

        Args:
            random_sample (Dataframe): The dataframe containing the random samples of entity-entity pairs that exist in KG

        Returns:
            int: the value of epsilon
        """

        return (1. / len(random_sample)) * np.sum(self.classify(random_sample))





if __name__ == "__main__":
    # root = os.getcwd()
    # reader = Reader()
    # reader.read_files(root + '/data/bbc/business/')
    g = G(["Trump demands trade war with China Hazzaaaa"])

