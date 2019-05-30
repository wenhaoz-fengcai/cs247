import unittest
from src.ee_graph import EE

NEWS = ["Trump demands trade war with China Hazzaaaa"]

class TestEE(unittest.TestCase):

    def test_init(self):
        self.ee = EE(NEWS)
        nodes = self.ee.get_nodes()
        assert set(nodes) == {"Trump", "China", "Hazzaaaa"}
        edges = self.ee.get_edges()
        assert set(edges) == {("Trump", "China"),
                              ("China", "Hazzaaaa")}
        df = self.ee.get_weights()
        assert df.loc["Trump", "China"] == 1
        assert df.loc["China", "Trump"] == 1
        assert df.loc["Hazzaaaa", "China"] == 1
        assert df.loc["China", "Hazzaaaa"] == 1

if __name__ == "__main__":
    unittest.main()