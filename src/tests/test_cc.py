import unittest
from src.cc_graph import CC

NEWS = ["Trump demands trade war with China Hazzaaaa"]

class TestCC(unittest.TestCase):

    def test_init(self):
        self.cc = CC(NEWS)
        nodes = self.cc.get_nodes()
        assert set(nodes) == {"demands", "trade", "war"}
        edges = self.cc.get_edges()
        assert set(edges) == {("demands", "trade"),
                              ("demands", "war"),
                              ("trade", "war")}

        df = self.cc.get_weights()
        assert df.loc["demands", "trade"] == 2
        assert df.loc["trade", "demands"] == 2
        assert df.loc["demands", "war"] == 2
        assert df.loc["war", "demands"] == 2
        assert df.loc["trade", "war"] == 2
        assert df.loc["war", "trade"] == 2

if __name__ == "__main__":
    unittest.main()