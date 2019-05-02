import unittest


class TestImports(unittest.TestCase):

    def test_imports(self):
        import pfb
        self.assertTrue(pfb.pfb_analysis.__name__ == "pfb_analysis")


if __name__ == "__main__":
    unittest.main()
