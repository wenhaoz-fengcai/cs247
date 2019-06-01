import pandas as pd
import numpy as np
import networkx as nx
import random
from random import randint
from src.reader import Reader
from src.search import Search
from src.graph import Graph
from src.ee_graph import EE
from src.ec_graph import EC
from src.cc_graph import CC
from src.kg_graph import KG
from numpy.linalg import inv


class Embedding(object):
    """

    """
    def __init__(self, d=10):

        self.read = Reader()
        self.search = Search()
        self.news_list = self.read.read_csv_file("../data/mixed-news/articles-title_only.csv")
        self.entity = self.read.parse_news(self.news_list[0:1])
        self.new, self.exist = self.search.query(self.entity)
        self.graph = Graph(self.news_list)
        self.words = self.graph.get_words()
        self.entities = self.graph.get_entities()
        self.ee_graph = EE(self.news_list)
        self.ec_graph = EC(self.news_list)
        self.cc_graph = CC(self.news_list)
        self.kg_graph = KG(self.news_list)
        self.S = pd.DataFrame(randint(0, 10), index=self.entities, columns=d)
        self.T = pd.DataFrame(randint(0, 10), index=self.words, columns=d)

    def weighted_sample(self, items, n):
        total = float(sum(w for w, v in items))
        i = 0
        w, v = items[0]
        while n:
            x = total * (1 - random.random() ** (1.0 / n))
            total -= x
            while x > w:
                x -= w
                i += 1
                w, v = items[i]
            w -= x
            yield v
            n -= 1

    def embedding_update(self, s, t, g, k=10):
        eta = 0.2
        # Sample an edge from G and draw k negative edges
        # and I guess, when we sample an edge, we also update that node's weight in the embedding!
        # So for sampling I should have all the weights,

        edges_weights = []
        edges = []
        warning = '!'
        sampled_edge = self.weighted_sample(g.get_edges(), 1)
        sampled_node_a = sampled_edge[0]
        sampled_node_b = sampled_edge[1]
        #swap!
        if s == 1 and t == 1:
            if type(sampled_node_a) == tuple:
                s1 = sampled_node_b
                sampled_node_b = sampled_node_a
                sampled_node_a = s1
        samplede_neg_nodes = []
        for i in k:
            if s == 1 and t == 1:
                samplede_neg_nodes[i] = g.get_entities()
            else:
                samplede_neg_nodes[i] = g.get_nodes()
        samplede_neg_nodes[k+1] = sampled_node_b
        # if s == 1 and t == 1:
        #     s1 = sampled_edge[0]
        #     s2 = sampled_edge[1]
        # if s == t:
        #     nodes = g.get_words()
        # else:
        #     nodes = g.get_nodes() # so I think we should have all the nodes! => get_nodes()
        # # draw k negative edges!
        # sampled_edge_nodes = random.sample(nodes, k) # [k]

        #so up until here, we have k negative edges, one positive edge, the graph, and S_t, T_t
        if s == 1 and t == 1:  # S, T, G_ec
            sum = 0
            for i in k+1:
                a = np.dot(self.S[sampled_edge[i]], self.T[sampled_node_a])
                b = np.exp(a)
                sum = sum + b
            c = np.log(sum)
            d = inv(self.S[sampled_node_b])
            e = - eta * d * c
            self.T[sampled_node_a] = self.T[sampled_node_a] - e
            sum = 0
            for i in k+1:
                a = np.dot(self.S[sampled_edge[i]], self.T[sampled_node_a])
                b = np.exp(a)
                sum = sum + b
            c = np.log(sum)
            d = inv(self.T[sampled_node_a])
            e = - eta * d * c
            self.S[sampled_node_b] = self.S[sampled_node_b] - e
        elif s == 0 and t == 1:  # T, T, G_cc
            sum = 0
            for i in k+1:
                a = np.dot(self.T[sampled_edge[i]], self.T[sampled_node_a])
                b = np.exp(a)
                sum = sum + b
            c = np.log(sum)
            d = inv(self.T[sampled_node_b])
            e = - eta * d * c
            self.T[sampled_node_a] = self.T[sampled_node_a] - e
        elif s == 1 and t == 0:  # S, S, G_ee
            sum = 0
            for i in k+1:
                a = np.dot(self.S[sampled_edge[i]], self.S[sampled_node_a])
                b = np.exp(a)
                sum = sum + b
            c = np.log(sum)
            d = inv(self.S[sampled_node_a])
            e = - eta * d * c
            self.S[sampled_node_b] = self.S[sampled_node_b] - e

    def joint_embedding(self):
        # I'll have the graph's, this is just initializing
        # graph_ee = nx.fast_gnp_random_graph(n=100, p=0.5)
        # graph_ec = nx.fast_gnp_random_graph(n=100, p=0.5)
        # graph_cc = nx.fast_gnp_random_graph(n=100, p=0.5)
        # graph_kg = nx.fast_gnp_random_graph(n=100, p=0.5)
        # the guiding parameter, which we should have empirically
        theta = 0.8
        # number of negative samplings
        k = 10
        # number of iterations
        t = 100
        # initializing the embeddings
        # Sampling would be like this: (though I'm not sure about the numbers)
        # S = pd.DataFrame(randint(0, 100), index=(0, 100), columns=(0, 100))
        # T = pd.DataFrame(randint(0, 100), index=(0, 100), columns=(0, 100))
        # the loop of the EM algorithm
        while t > 0:
            gamma = random.uniform(0, 1)
            if gamma <= theta:
                self.embedding_update(1, 0, self.kg_graph, k)
            else:
                self.embedding_update(1, 1, self.ec_graph, k)
                self.embedding_update(1, 0, self.ee_graph, k)
                self.embedding_update(0, 1, self.cc_graph, k)
            t = t - 1
