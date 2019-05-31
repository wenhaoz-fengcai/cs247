import unittest
from src.ec_graph import EC

NEWS = ["Trump demands hazzaaaa"]

class TestEC(unittest.TestCase):

    def test_init(self):
        self.ec = EC(NEWS, window_size=3)
        nodes = self.ec.get_nodes()
        assert set(nodes) == {"Trump", "demands", "hazzaaaa"}
        edges = self.ec.get_edges()
        assert set(edges) == {("Trump", "demands"),
                              ("Trump", "hazzaaaa")}
        df = self.ec.get_weights()
        print(df)
        assert df.loc["Trump", "demands"] == 1
        assert df.loc["demands", "Trump"] == 1
        assert df.loc["Trump", "hazzaaaa"] == 1
        assert df.loc["hazzaaaa", "Trump"] == 1
        assert df.loc["demands", "hazzaaaa"] == 0
        assert df.loc["hazzaaaa", "demands"] == 0

if __name__ == "__main__":
    unittest.main()