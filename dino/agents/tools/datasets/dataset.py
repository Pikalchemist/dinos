'''
    File name: dataset.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from dino.data.spacemanager import SpaceManager


class Dataset(SpaceManager):
    """The dataset used to record actions, outcomes and procedures."""

    def __init__(self, options={}):
        """
        options dict: parameters for the dataset
        """
        super().__init__(storesData=True)
        self.options = options

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['options']))
        return dict_
