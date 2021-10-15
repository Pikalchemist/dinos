import unittest
from pathlib import Path


from dinos.data.spacemanager import SpaceManager
from dinos.data.space import Space


class TestLoader(unittest.TestCase):
    def test_space(self):
        sm = SpaceManager('test')

        A = Space(sm, 2)
        B = Space(sm, 1)

        print(A.zero())
        print(A.point([1, 2]))


if __name__ == '__main__':
    unittest.main()
