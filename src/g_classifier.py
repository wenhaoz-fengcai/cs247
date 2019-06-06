import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from src.ee_graph import EE
from src.kg_graph import KG
from src.reader import Reader
from src.joint_embedding import Embedding


class G_Classifier():
    """
    The class of the classifier g. This class first embeds all entity-entity pairs and then models P(z=1|x) from the
    KG and uses this to estimate P(z=1|y=1). The classifier f is then derived from this classifier.

    Attributes:
        ee_graph: The entity-entity graph
        classifier: The classifier used to model P(z=1|x). A neural network.
        embeddings: The embeddings of all words in the KG
        kg_df: The combinations of known pairs of entities and their embedings
        new_comb: The combinations of new pairs of entities
        all_df: The combinations of all pairs of entities and their embedings and 'z' values
        S_known: The random subset of all_df which contains entity pairs that are known
        eps: The value of epsilon which is used to scale the classifier. Estiamtion of P(z=1|y=1).
        results_df: The pairs of all e-e pairs and their corresponding probability of being an emerging relation
        emerging_relations: The predict emerging relations after filtering results_df
    """

    def __init__(self, architecture=(10,10,10), sample_size=0.25, threshold=0.91):
        """
        Constructor

        Args:
            lst_news (list): The news to parse
            architecture (tuple, optional): The architecture of the hidden layers in the classifier. Defaults to (10,10,10).
            sample_size (float, optional): The fraction of total samples to randomly sample in order to estimate epsilon. Defaults to 0.25.
            threshold (float, optional): The probability threshold to be considered an emerging relation. Defaults to 0.7.
        """
        # self.embedder = Embedding()
        # self.ee_graph = self.embedder.ee_graph
        # self.kg_graph = self.embedder.kg_graph
        self.classifier = MLPClassifier(hidden_layer_sizes=architecture)
        self.classifier = LogisticRegression()

        # self.embeddings = self.embedder.joint_embedding()[0]
        self.embeddings, self.entities, self.kg, self.new = self.parse_csv()
        # self.all_df, self.kg_df, self.new_comb = self.create_combinations(self.ee_graph.get_nodes, self.kg_graph.get_nodes, self.ee_graph.nodes, self.embeddings)
        self.all_df, self.kg_df, self.new_comb = self.create_combinations(self.entities, self.kg, self.new, self.embeddings)
        self.S_known = self.random_sample(self.all_df, sample_size)
        self.classifier = self.train_classifier(self.classifier, self.all_df)
        self.eps = self.generate_eps(self.classifier, self.S_known)
        self.results_df = self.predict_emerging_probs(self.classifier, self.all_df, self.eps)
        self.emerging_relations = self.filter_results(self.results_df, self.kg_df, self.new_comb, threshold)

    def parse_csv(csv):
        with open('entities.csv', 'r') as f:
            embeddings = {}
            entities = []
            f.readline()
            for line in f.readlines():
                line = line.split(',')
                embeddings[line[0]] = [line[1:]]
                entities.append(line[0])
        kg = ['Britain', 'Clarke', 'Kingdom', 'Michael', 'Rome', 'Straw', 'Telegraph', 'YouGov'] # np.unique(np.random.choice(entities, size=int(len(entities) * 0.5)))
        embeddings = pd.DataFrame.from_dict(embeddings, orient='index')
        new = np.setdiff1d(np.array(entities), kg)

        entities = [(x, []) for x in entities]
        kg = [(x, []) for x in kg]
        new = [(x, []) for x in new]

        return embeddings, entities, kg, new


    def embed_pairs(self, pairs):
        """
        Embeds an entity-entity pair according to the function h and combines the pair and embedding into a dataframe
        indexed by the pair

        Args:
            pairs (list): A list containing entity-entity pairs

        Returns:
            Dataframe: A df containing the embeddings of the entity-entity pairs as values and indexed by e-e pair
        """


        embeddings = {str(pair[0]): [self.h(pair[1][0], pair[1][1])] for pair in pairs}
        embeddings = pd.DataFrame.from_dict(embeddings, orient='index')
        embeddings = embeddings.rename(columns={0: 'embedding'})

        return embeddings

    def create_combinations(self, ee_nodes, kg_nodes, nodes, embeddings):
        """
        Top level function that parses entities from graphs, pairs them with their embedings,
        calls embed_pairs() to combine embedings, and gives proper 'z' labels to all e-e pairs.

        Args:
            ee_nodes (list): A list containing all entities as (word, label) pairs
            kg_nodes (list): A list containing known entites as (word, label) pairs
            nodes (dict): A dict of all new nodes in ee graph
            embeddings (dataframe): The embeddings of each word indexed by the enitity

        Returns:
            dataframe: Containing all combiantions of entity-entity pairs with embeddings and z labels
            dataframe: Containing all combiantions of known entity-entity pairs with embeddings
            list: Containing all combinations of new entity-entity pairs
        """

        # parse entities from list
        all_entities = list(zip(*ee_nodes))[0]
        kg_entities = list(zip(*kg_nodes))[0]
        # new_entities = list(zip(*nodes['N']))[0]
        new_entities = list(zip(*nodes))[0]

        # create all combinations of entity-entity pairs
        all_comb = combinations(all_entities, 2)
        kg_comb = combinations(kg_entities, 2)
        new_comb = combinations(new_entities, 2)

        # pair the entity-entity pairs with their embeddings
        all_comb = [(pair, (embeddings.loc[pair[0], 0], embeddings.loc[pair[1], 0])) for pair in all_comb]    # dont know column name in dataframe corresponding to embedding yet
        kg_comb = [(pair, (embeddings.loc[pair[0], 0], embeddings.loc[pair[1], 0])) for pair in kg_comb]    # dont know column name in dataframe corresponding to embedding yet

        # combine entity-entity pair embedings together
        all_df = self.embed_pairs(all_comb)
        kg_df = self.embed_pairs(kg_comb)

        # give proper "z" labels to entity-entity pairs that exist in KG
        all_df.insert(1, 'z', 0)
        intersection = all_df.index.intersection(kg_df.index)
        all_df.loc[intersection, 'z'] = 1

        return  all_df, kg_df, new_comb

    def h(self, x, y):
        """
        Combines two word embeddings according to function defined in the paper (average)

        Args:
            x (np array): The embedding of the first word
            y (np array): The embedding of the second word

        Returns:
            np array: The embedding of the entity-entity pair
        """

        x = [float(n) for n in x]
        y = [float(n) for n in y]
        return 0.5 * np.add(x, y) # forumula from paper

    def random_sample(self, all_df, sample_size):
        """
        Obtain random sample of all entity pairs which are in the KG

        Args:
            pairs (Dataframe): The dataframe containing the embeddings of all entity-entity pairs
            sample_size (int): The size of the sample to take from the dataframe

        Returns:
            Dataframe: The subset of entity-entity pairs from the random sample that are in the KG
        """

        sample_size = int(all_df.shape[0] * sample_size)
        S = all_df.sample(n=sample_size)
        S_known = S.where(S['z'] == 1).dropna()

        return S_known

    def train_classifier(self, classifier, all_df):
        """
        For training the classifier

        Args:
            classifier (sklearn classifier): The classifier to train
            pairs (dataframe): The dataframe containing all entity pairs

        Returns:
            sklearn classifier: The trained classifier
        """

        X = all_df.loc[:, "embedding"].as_matrix()
        X = np.array([x.tolist() for x in X])
        y = all_df.loc[:, 'z'].as_matrix()

        classifier.fit(X, y)

        return classifier

    def generate_eps(self, classifier, random_sample):
        """
        For calculating epsilon which is used to scale the classifier g to get final predictions

        Args:
            classifier (sklearn classifier): The trained classifier
            random_sample (Dataframe): The dataframe containing the random samples of entity-entity pairs that exist in KG

        Returns:
            float: the value of epsilon
        """

        X = random_sample.loc[:, "embedding"].as_matrix()
        X = np.array([x.tolist() for x in X])
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

        X = all_df.loc[:, 'embedding'].as_matrix()
        X = np.array([x.tolist() for x in X])
        probs = classifier.predict_proba(X)
        probs = probs / eps
        results_df = all_df.copy()
        results_df.insert(2, 'probs', probs[:, 0])
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
        intersection = results_df.index.difference(kg_df.index)
        results_df = results_df.loc[intersection, :]

        # get entity-entity pairs that have at least 1 node in KG
        new_comb = pd.Index(new_comb)
        intersection = results_df.index.difference(new_comb)
        # print(intersection)
        results_df = results_df.loc[intersection, :]

        # print(results_df)
        # filter results down to entity-entity pairs with probability >= threshold
        results_df = results_df.where(results_df['probs'] >= threshold).dropna()

        return results_df








if __name__ == "__main__":
    g = G_Classifier()
    print('kg\n', [x[0] for x in g.kg])
    print(list(g.emerging_relations.index))