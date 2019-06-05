import pandas as pd
import numpy as np
import random
import csv
from random import randint
from src.reader import Reader
from src.graph import Graph
from src.ee_graph import EE
from src.ec_graph import EC
from src.cc_graph import CC
from src.kg_graph import KG
from numpy.linalg import inv


class Embedding(object):
        """
        Python class which produces the joint embedding of the words and entities.
        To use it: E = Embedding(), entity_embedding, word_embedding = E.joint_embedding()
        Attributes:
        kg_graph: The knowledge Graph
        ee_graph: A heterogeneous subgraph of HEER, showing relations between entities
        cc_graph: A heterogeneous subgraph of HEER, showing relations between words
        ec_graph: A bipartite subgraph of HEER, showing relations between entities and words
        """
        def __init__(self):

        self.read = Reader()
#           self.news_list = ["Today's policy is about global warming", "Donald Trupm is the president of United States", "UCLA is the best school in southern California", "Noor Nakhaei is going to be student at UCLA", "the Boelter Hall is a dungeon", "UCLA is colaborating with Stanford", "Wenhao is meeting Trump", "Trump is in United Kingdom"]
          self.news_list = self.read.read_csv_file("./data/mixed-news/articles-title_only.csv")
          self.graph = Graph(self.news_list)
          self.words = self.graph.get_words()
          self.entities = self.graph.get_entities()
          self.ee_graph = EE(self.news_list)
          self.ec_graph = EC(self.news_list)
          self.cc_graph = CC(self.news_list)
          print("cc", self.cc_graph.get_edges())
          self.kg_graph = KG(self.news_list)
          self.d = 10 #THIS SHOULD BE CHANGED! 4, 10, 18  
          self.S = pd.DataFrame(1, index=self.entities, columns=range(0, self.d))
          self.T = pd.DataFrame(1, index=self.words, columns=range(0, self.d))
          for i in self.S.columns:
            for j in self.S.index:
              self.S[i][j] = randint(0, 10)
          for i in self.T.columns:
            for j in self.T.index:
              self.T[i][j] = randint(0, 10)

        def weighted_sample(self, items, n):
          """
          This function samples an item, proportional to it's weight attribute.
          Args:
              items: the list of edges we should choose between them.
              n: number of edges we should choose.
          Returns:
              Yields the chosen edge, proportional to it' weight
          """
          total = 0
          for j in items:
            total = float(sum(w for a, b, w in items))
              
          i = 0
          a, b, w = items[0]
          while n:
              x = total * (1 - random.random() ** (1.0 / n))
              total -= x
              while x > w:
                  x -= w
                  i += 1
                  a, b, w = items[i]
              w -= x
              yield a, b
              n -= 1

        def embedding_update(self, s, t, g, k=3):
          """
          This function updates the embeddings of words and entitites.
          Args:
              s: A binary number, indicting the type of embedding that should be updated.
              t: A binary number, indicting the type of embedding that should be updated.
              g: The graph; It could be the ee, cc, or ec subgraph, or the kg graph.
              k: Number of negative edges.

          """
          eta = 0.2       
          # Sample an edge from G and draw k negative edges
          # and I guess, when we sample an edge, we also update that node's weight in the embedding!
          # So for sampling I should have all the weights,
          df = g.get_weights()
          num_cols = g.get_nodes()
          edges = []
          for i in num_cols:
            for j in num_cols:
              if df[i][j] != 0:
                edge = []
                edge.append(i)
                edge.append(j)
                edge.append(df[i][j])
                edges.append(edge)
          sampled_edge = self.weighted_sample(edges, 1)

          for el in sampled_edge:
            sampled_node_a = el[0]
            sampled_node_b = el[1]

          #swap!
          if s == 1 and t == 1:
              print(sampled_node_a)
              print(sampled_node_b)
              if sampled_node_a in self.S.index:
                  s1 = sampled_node_b
                  sampled_node_b = sampled_node_a
                  sampled_node_a = s1
          # sampled_neg_nodes = []
          if s == 1 and t == 1:
              nodes = g.get_entities()
          else:
              nodes = g.get_nodes()
          # draw k negative edges!
          sampled_neg_nodes = random.sample(nodes, k) # [k]
          sampled_neg_nodes.append(sampled_node_b)

          #so up until here, we have k negative edges, one positive edge, the graph, and S_t, T_t
          if s == 1 and t == 1:  # S, T, G_ec
              sum = 0
              for i in range(k+1):
                  a = np.dot(self.S.loc[sampled_neg_nodes[i]], self.T.loc[sampled_node_a])
                  if a > 123:
                    a = 123
                  elif a < 0.1:
                    a = 0.5
                  b = np.exp(a)
                  sum = sum + b
              c = np.log(sum)
              d = self.S.loc[sampled_node_b].T
              e = - eta * d * c
              self.T.loc[sampled_node_a] = self.T.loc[sampled_node_a] - e
              sum = 0
              for i in range(k+1):
                  a = np.dot(self.S.loc[sampled_neg_nodes[i]], self.T.loc[sampled_node_a])
                  if a > 123:
                    a = 123
                  elif a < 0.1:
                    a = 0.5
                  b = np.exp(a)
                  sum = sum + b
              c = np.log(sum)
              d = self.T.loc[sampled_node_a].T

              e = - eta * d * c
              self.S.loc[sampled_node_b] = self.S.loc[sampled_node_b] - e
          elif s == 0 and t == 1:  # T, T, G_cc
              sum = 0
              for i in range(k+1):
                  a = np.dot(self.T.loc[sampled_neg_nodes[i]], self.T.loc[sampled_node_a])
                  if a > 123:
                    a = 123
                  elif a < 0.1:
                    a = 0.5
                  b = np.exp(a)
                  sum = sum + b
              c = np.log(sum)
              d = self.T.loc[sampled_node_b].T
              e = - eta * d * c
              self.T.loc[sampled_node_a] = self.T.loc[sampled_node_a] - e
          elif s == 1 and t == 0:  # S, S, G_ee
              sum = 0
              for i in range(k+1):
                  a = np.dot(self.S.loc[sampled_neg_nodes[i]], self.S.loc[sampled_node_a])
                  if a > 123:
                    a = 123
                  elif a < 0.1:
                    a = 0.5
                  b = np.exp(a)
                  sum = sum + b
              c = np.log(sum)
              d = self.S.loc[sampled_node_a].T
              e = - eta * d * c
              self.S.loc[sampled_node_b] = self.S.loc[sampled_node_b] - e

        def joint_embedding(self):
          """
          This function runs the iteration to minimize the cost function, and calls the update function..
          Attributes:
              theta: The guiding parameter, chosen empirically. The bigger it is, the more effective the kg graph is.
              k: Number of negatve smaples.
              t: Number of iterations.
          Returns:
              Returns two dataframes, first the entitiy embedding(normalized_S) and second the word embedding(normalized_T).

          """
          # the guiding parameter, which we should have empirically, the bigger it is, the more we are relying to our kg graph.
          theta = 0.5 #THIS SHOULD BE CHANGED! 0.2, 0.5, 0.7
          # number of negative samplings
          k = 2
          # number of iterations
          t = 100
          # the loop of the algorithm
          while t > 0:
              gamma = random.uniform(0, 1)
              if gamma <= theta:
                  self.embedding_update(1, 0, self.kg_graph, k)
              else:
                  self.embedding_update(1, 1, self.ec_graph, k)
                  self.embedding_update(1, 0, self.ee_graph, k)
                  self.embedding_update(0, 1, self.cc_graph, k)
              t = t - 1
          normalized_S=self.S.div(self.S.sum(axis=1), axis=0)
          normalized_T=self.T.div(self.T.sum(axis=1), axis=0)
          return normalized_S, normalized_T
#           print(normalized_S, normalized_T)
#           normalized_S.to_csv('S', sep='\t')
#           normalized_T.to_csv('T', sep='\t')
