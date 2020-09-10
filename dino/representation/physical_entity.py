from .live_entity import LiveEntity


class PhysicalEntity(LiveEntity):
    PHYSICAL = True

    def __init__(self, kind, name='', spaceManager=None):
        super().__init__(kind, name, spaceManager=spaceManager)
        self.shape = None

    def _activate(self):
        super()._activate()
        self.initPhysics(self.engine.physics)

    def _deactivate(self):
        super()._deactivate()
        self.stopPhysics(self.engine.physics)

    def initPhysics(self, physics):
        return None

    def stopPhysics(self, physics):
        pass
