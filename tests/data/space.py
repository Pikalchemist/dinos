import unittest
from pathlib import Path


from dino.data.spacemanager import SpaceManager
from dino.data.space import Space


class TestLoader(unittest.TestCase):
    def test_space(self):
        sm = SpaceManager()

        A = Space(sm, 2)
        B = Space(sm, 1)

        print(A.zero())
        print(A.point([1, 2]))


if __name__ == '__main__':
    unittest.main()
