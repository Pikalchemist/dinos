from ..engine import Engine


class InternalEngine(Engine):
    PHYSICAL = True

    # Physics
    def updatePhysics(self, dt=0.00000001):
        self._updatePhysics(dt)
        self.iteration += 1
        self.time += dt

    def _updatePhysics(self, dt):
        pass
