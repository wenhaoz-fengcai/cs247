import unittest
from src.graph import Graph

NEWS = ["President Trump took a few sharp verbal shots at House Speaker Nancy Pelosi at the White House Thursday as he announced $16 billion in new subsidies for farmers and ranchers, questioning the mental faculties of the most powerful woman in Washington.", "Hazzaaaa!"]

NEWS_NODES = {'N': [('Hazzaaaa', 'O')], 'C': [('President', 'O'), ('Trump', 'PERSON'), ('took', 'O'), ('sharp', 'O'), ('verbal', 'O'), ('shots', 'O'), ('House', 'ORGANIZATION'), ('Speaker', 'O'), ('Nancy', 'PERSON'), ('Pelosi', 'PERSON'), ('White', 'ORGANIZATION'), ('House', 'ORGANIZATION'), ('Thursday', 'O'), ('announced', 'O'), ('billion', 'O'), ('new', 'O'), ('subsidies', 'O'), ('farmers', 'O'), ('ranchers', 'O'), ('questioning', 'O'), ('mental', 'O'), ('faculties', 'O'), ('powerful', 'O'), ('woman', 'O'), ('Washington', 'LOCATION')]}

NEWS_WORDS = ['Hazzaaaa', 'President', 'Trump', 'took', 'sharp', 'verbal',
              'shots', 'House', 'Speaker', 'Nancy', 'Pelosi', 'White',
              'House', 'Thursday', 'announced', 'billion', 'new', 'subsidies',
              'farmers', 'ranchers', 'questioning', 'mental', 'faculties', 'powerful', 'woman', 'Washington']

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(NEWS)


    def test_get_words(self):
        self.graph = Graph(NEWS)
        res = self.graph.get_words()

        assert set(res) == set(NEWS_WORDS)

    def test_sliding_window(self):
        a=[1,2,3,4,5,6,7]
        it = self.graph.sliding_window(a, 5)
        res = []
        for i in it:
            res.append(i)
        assert res == [(1, 2, 3, 4, 5),
                       (2, 3, 4, 5, 6),
                       (3, 4, 5, 6, 7)]

    def test_get_edges(self):
        self.graph = Graph(["text graph edge node link window"])
        res = self.graph.get_edges()
        assert res == {('edge', 'node'), ('text', 'link'), ('graph', 'window'), ('text', 'node'), ('edge', 'link'), ('node', 'link'), ('node', 'window'), ('graph', 'link'), ('text', 'edge'), ('graph', 'edge'), ('text', 'graph'), ('graph', 'node'), ('edge', 'window'), ('link', 'window')}

    def test_get_weights1(self):
        self.graph = Graph(["text edge graph edge"], 3)
        df = self.graph.get_weights()

        assert df.loc["text", "graph"] == 1
        assert df.loc["graph", "text"] == 1
        assert df.loc["text", "edge"] == 1
        assert df.loc["edge", "text"] == 1
        assert df.loc["edge", "graph"] == 3
        assert df.loc["graph", "edge"] == 3

    def test_get_weights2(self):
        self.graph = Graph(["text edge graph edge",
                            "text edge graph edge"], 3)
        df = self.graph.get_weights()

        assert df.loc["text", "graph"] == 2
        assert df.loc["graph", "text"] == 2
        assert df.loc["text", "edge"] == 2
        assert df.loc["edge", "text"] == 2
        assert df.loc["edge", "graph"] == 6
        assert df.loc["graph", "edge"] == 6


if __name__ == "__main__":
    unittest.main()