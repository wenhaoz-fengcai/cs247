import unittest
from src.ec_graph import EC

NEWS = ["Trump demands Hazzaaaa"]

class TestEC(unittest.TestCase):

    def test_init(self):
        self.ec = EC(NEWS, window_size=2)
        nodes = self.ec.get_nodes()
        assert set(nodes) == {"Trump", "demands", "Hazzaaaa"}
        edges = self.ec.get_edges()
        assert set(edges) == {("Trump", "demands"),
                              ("demands", "Hazzaaaa")}
        df = self.ec.get_weights()
        assert df.loc["Trump", "demands"] == 1
        assert df.loc["demands", "Trump"] == 1
        assert df.loc["Hazzaaaa", "demands"] == 1
        assert df.loc["demands", "Hazzaaaa"] == 1

if __name__ == "__main__":
    unittest.main()