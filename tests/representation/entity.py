import unittest
from pathlib import Path


from dino.data.spacemanager import SpaceManager
from dino.representation.entity import Entity
from dino.representation.property import Property


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
