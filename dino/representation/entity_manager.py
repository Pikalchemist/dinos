from exlab.interface.serializer import Serializable

from dino.representation.entity import Entity, ProxyEntity


class EntityManager(object):
    def __init__(self, name, entityCls=Entity, proxyCls=ProxyEntity):
        self.name = name
        self.entityCls = entityCls
        self.proxyCls = proxyCls

        self.engine = None
        self.world = self.entityCls('root', manager=self)

        self.conserveRoot = False

    def __repr__(self):
        return f'EntityManager({len(self.world.cascadingChildren()) + 1} entities)'
    
    def findEntity(self, name):
        return self.world.cascadingChild(name)

    def findProperty(self, name):
        return self.world.cascadingProperty(name)
    
    def entities(self):
        return [self.world] + self.world.cascadingChildren()
    
    def agents(self):
        return [host.hosting for host in self.world.hosts() if host.hosting]

    def scheduledActions(self, notNone=True):
        if notNone:
            return {agent: agent.scheduled for agent in self.agents() if agent.scheduled}
        return {agent: agent.scheduled for agent in self.agents()}

    def countScheduledActions(self):
        return len(self.scheduledActions())

    def clear(self):
        self.world.clearChildren()
    
    def propertySpace(self, filterOrProperty=None, kind=None, dataset=None):
        if isinstance(filterOrProperty, str):
            filterOrProperty = self.world.cascadingProperty(filterOrProperty)
        space = filterOrProperty.space
        if dataset:
            space = space.convertTo(spaceManager=dataset, kind=kind)
        return space

    # Entities
    def convertEntity(self, entity):
        relatedEntity = next((e for e in self.entities() if e.linkedTo(entity)), None)
        if relatedEntity:
            return relatedEntity

        entity = entity.createLinkedEntity(self)
        if entity.isRoot() and not self.conserveRoot:
            self.world = entity
        elif not entity.parent:
            self.world.addChild(entity)
        entity = self.customEntityConversion(entity)
        return entity

    def customEntityConversion(self, entity):
        return entity
