import unittest
from pathlib import Path


from dinos.data.spacemanager import SpaceManager
from dinos.representation.entity import Entity
from dinos.representation.property import Property


class TestEntity(unittest.TestCase):
    def test_simple(self):
        sm = SpaceManager()

        obj1 = Entity('box')
        sm.scene.addChild(obj1)

        color = Property(obj1, 'color', 1)

        print(sm.scene.cascadingChildren())
        print(sm.scene.cascadingProperties())


if __name__ == '__main__':
    unittest.main()
