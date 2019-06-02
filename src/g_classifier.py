import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.ee_graph import EE
from src.kg_graph import KG
from src.reader import Reader


class G_Classifier():
    """
    The class of the initial classifier g. This class first embeds all entity-entity pairs and then models z from the
    KG. The classifier f is then derived from this classifier.

    Attributes:
        ee_graph: The entity-entity sub-graph of the KG
        classifier: The classifier used to model P(z=1|y=1). A neural network
        embeddings: The embeddings of all words in the KG
        known_comb: The combinations of known pairs of entities
        new_comb: The combinations of new pairs of entities
        all_comb: The combinations of all pairs of entities
        S_known: The random subset of all_comb which contains entity pairs that are known
        eps: The value of epsilon which is used by the second classifier
    """

    def __init__(self, lst_news, architecture=(10,10,10), sample_size=0.25, minimum_sample=50, threshold=0.7):

        self.ee_graph = EE(lst_news)
        self.kg_graph = KG(lst_news)
        self.classifier = MLPClassifier(hidden_layer_sizes=architecture)

        self.embeddings = None #call function to get embeddings
        self.all_df, self.kg_df self.new_comb = __create_combinations(self.ee_graph.get_nodes, self.kg_graph.get_nodes, self.ee_graph.nodes, self.embeddings)
        self.S_known = self.__random_sample(self.all_df, sample_size, minimum_sample)
        self.classifier = __train_classifier(self.classifier, self.all_df)
        self.eps = self.__generate_eps(self.S_known)
        self.results_df = self.predict_emerging_probs(self.classifier, self.eps)
        self.emerging_relations = self.filter_results(self.results_df, self.kg_df, self.new_comb, threshold)

    def __embed_pairs(self, pairs):
        """
        Private class method that embeds an entity-entity pair according to the function h.

        Args:
            pairs (list): A list containing entity-entity pairs

        Returns:
            Dataframe: A df containing the embeddings of the entity-entity pairs as values and indexed by e-e pair
        """

        embeddings = {str(pair[0]), self.__h(pair[1][0], pair[1][0]) for pair in pairs}
        embeddings = pd.from_dict(embeddings, orient='index')
        embeddings.rename(index=str, columns={'0': 'embedding'})
        return embeddings

    def __create_combinations(self, ee_nodes, kg_nodes, nodes, embeddings):
        """
        Private class method. Top level function that parses entities from graphs, pairs them with their embedings,
        calls __embed_pairs() to combine embedings, and gives proper 'z' labels to all e-e pairs.

        Args:
            ee_nodes (list): A list containing all entities as (word, label) pairs
            kg_nodes (list): A list containing known entites as (word, label) pairs
            nodes (dict): A dict of all all nodes in ee graph
            embeddings (dataframe): The embeddings of each word indexed by the enitity

        Returns:
            dataframe: Containing all combiantions of entity-entity pairs with embeddings and z labels
            dataframe: Containing all combiantions of known entity-entity pairs with embeddings
            list: Containing all combinations of new entity-entity pairs
        """

        # parse entities from list
        all_entities = list(zip(*ee_nodes))[0]
        kg_entities = list(zip(*kg_nodes))[0]
        new_entities = list(zip(*nodes['N']))[0]

        # create all combinations of entity-entity pairs
        all_comb = combinations(all_entities, 2)
        kg_comb = combinations(kg_entities, 2)
        new_comb = combinations(new_entities, 2)

        # pair the entity-entity pairs with their embeddings
        all_comb = [(pair, (embeddings.loc[pair[0], "0"], embeddings.loc[pair[1], "0"])) for pair in all_comb]    # dont know column name in dataframe corresponding to embedding yet
        kg_comb = [(pair, (embeddings.loc[pair[0], "0"], embeddings.loc[pair[1], "0"])) for pair in kg_comb]    # dont know column name in dataframe corresponding to embedding yet

        # combine entity-entity pair embedings together
        all_df = self.__embed_pairs(all_comb)
        kg_df = self.__embed_pairs(kg_comb)

        # give proper "z" labels to entity-entity pairs that exist in KG
        all_df.insert(1, 'z', 0)
        intersection = all_df.Index.intersection(kg_df.Index)
        all_df.loc[intersection, 'z'] = 1

        return  all_df, kg_df, new_comb

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

    def __random_sample(self, all_df, sample_size, min_sample):
        """
        Obtain random sample of all entity pairs which are in the KG

        Args:
            pairs (Dataframe): The dataframe containing the embeddings of all entity-entity pairs
            sample_size (int): The size of the sample to take from the dataframe

        Returns:
            Dataframe: The subset of entity-entity pairs from the random sample that are in the KG
        """

        sample_size = all_df.shape[0] * sample_size
        size_known = 0
        while size_known < min_sample:  # want a random sample of minimum size (min size is arbitrary)
            S = all_df.sample(sample_size)
            S_known = S.where(S['z'] == 1)
            size_known = S_known.shape[0]

        return S_known

    def __train_classifier(self, classifier, all_df):
        """
        private class method for training the classifier

        Args:
            classifier (sklearn classifier): The classifier to train
            pairs (dataframe): The dataframe containing all entity pairs

        Returns:
            sklearn classifier: The trained classifier
        """

        X = np.array(all_df.loc[:, "embedding"])
        y = np.array(all_df.loc[:, 'z'])

        classifier.fit(X, y)

        return classifier

    def __generate_eps(self, classifier, random_sample):
        """
        Private class method for calculating epsilon which is used to scale the classifier f

        Args:
            classifier (sklearn classifier): The trained classifier
            random_sample (Dataframe): The dataframe containing the random samples of entity-entity pairs that exist in KG

        Returns:
            float: the value of epsilon
        """

        X = np.array(random_sample.loc[:, "embedding"])
        size = random_sample.shape[0]
        probs = classifier.predict_proba(X)

        return (1. / size) * np.sum(probs)

    def predict_emerging_probs(self, classifier, all_df, eps):
        """
        Predict the probability of each entity-entity pair being an emerging relation.

        Args:
            classifier (sklearn classifier): The trained classifier
            eps (float): The value of epsilon
        """

        X = np.array(all_df.loc[:, 'embedding'])
        probs = classifier.predict_proba(X)
        probs = probs / eps
        results_df = all_df.copy()
        results_df.insert(2, 'probs', probs)
        return results_df

    def filter_results(self, results_df, kg_df, new_comb, threshold):
        """
        Filter results down to entity-entity pairs that have exactly one node in KG and whose probabilities are >=
        threshold

        Args:
            results_df (Dataframe): The dataframe containing predicted probabilities for each e-e pair
            kg_df (Dataframe): The KG dataframe
            new_comb (list): The list containing all combinations of new entity-entity pairs
            threshold (float): The theshold the probability must be greater than in order to be consider an emerging relation

        Returns:
            Dataframe: The remaining relations
        """

        # get entity-entity pairs that are not in KG
        intersection = results_df.Index.intersection(kg_df.Index)
        results_df = results_df.loc[intersection, :]

        # get entity-entity pairs that have at least 1 node in KG
        new_comb = pd.Index(new_comb)
        intersection = results_df.Index.intersection(new_comb)
        results_df = results_df.loc[intersection, :]

        # filter results down to entity-entity pairs with probability >= threshold
        results_df = results_df.where(results_df['probs'] >= threshold)

        return results_df








if __name__ == "__main__":
    # root = os.getcwd()
    # reader = Reader()
    # reader.read_files(root + '/data/bbc/business/')
    g = G(["Trump demands trade war with China Hazzaaaa"])
