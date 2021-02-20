'''
    File name: dataset.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from exlab.modular.module import manage

from dino.data.spacemanager import SpaceManager


class Dataset(SpaceManager):
    """The dataset used to record actions, outcomes and procedures."""

    def __init__(self, options={}):
        """
        options dict: parameters for the dataset
        """
        super().__init__('dataset', storesData=True, options=options)
        self.logger.tag = 'dataset'
        self.learner = None

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['options']))
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(dict_.get('options', {}))
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        serializer.set('dataset', self)
        serializer.set('dataset', self, category='spaceManager')
    
    def attachLearner(self, learner):
        self.learner = learner
        manage(self).attach(learner)
