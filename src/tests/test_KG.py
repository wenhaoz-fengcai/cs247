import unittest
from src.kg_graph import KG

NEWS = ["Trump demands trade war with China"]

class TestKG(unittest.TestCase):

    def test_init(self):
        self.kg = KG(NEWS)
        nodes = self.kg.get_nodes()
        assert set(nodes) == {"Trump", "China"}
        edges = self.kg.get_edges()
        assert set(edges) == {("Trump", "China")}
        df = self.kg.get_weights()
        assert df.loc["Trump", "China"] == 1
        assert df.loc["China", "Trump"] == 1

if __name__ == "__main__":
    unittest.main()