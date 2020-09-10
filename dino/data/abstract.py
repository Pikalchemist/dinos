from .space import SpaceKind


class AMTBase(object):
    NAME = 'Base'

    def __init__(self, context, parent=None, element=None):
        self.context = context
        self.element = element
        self.parents = set()
        self.children = set()
        # self.addParent(parent)

    @property
    def abstract(self):
        return True

    def copy(self):
        obj = self.__class__.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.children = [child.copy() for child in obj.children]
        return obj

    def addChild(self, child):
        if child not in self.children:
            self.children.add(child)
            child.parent = self

    def addChildren(self, children):
        for child in children:
            self.addChild(child)

    # def addParent(self, parent):
    #     if not parent:
    #         return
    #     if parent not in self.parents:
    #         self.parents.add(parent)
    #         parent.addChild(self)

    # Visitor
    def abstractElements(self):
        elements = set()
        if self.abstract and self.element:
            elements.add(self.element)
        elements |= set(
            e for child in self.children for e in child.abstractElements())
        return elements

    def assignableElements(self):
        assignables = {}
        for child in self.children:
            assignables.update(child.assignableElements())
        if self.abstract and self.element:
            assignables[self.element] = set(self.element.assignables)
        return assignables

    def assign(self, element, value):
        tree = self.copy()
        tree.assignInplace(element, value)
        return tree

    def assignInplace(self, element, value):
        if self.element == element:
            self.element = AMTElement(
                self.context, self.__class__, assigned=value)
        for child in self.children:
            child.assignInplace(element, value)

    def get(self, spaceManager):
        pass

    # Representation
    def __repr__(self):
        return self.toStr()

    def toStr(self):
        return '{}'.format(', '.join([item.toStr() for item in self.children]))

    def name(self):
        if self.element:
            return self.element.name()
        raise Exception('Not a namable abstract object!')

    def fullname(self):
        return self._fullnamePattern(self.name())

    @staticmethod
    def _namePattern(obj, index):
        raise Exception('Not a namable abstract object!')

    def _fullnamePattern(self, name):
        raise Exception('Not a namable abstract object!')


class AMT(AMTBase):
    def __init__(self, context=None):
        super().__init__(context if context else self)

        self.fullnames = {-1: '__base__'}
        self.names = {-1: '__base__'}
        self.matching = {}
        self.sameRowsSpaces = True

        if context:
            context.addChild(self)

    def expr(self, item):
        obj = AMTExpr(self.context, item)
        self.addChild(obj)
        return obj

    def entity(self, assigned=None, element=None):
        obj = AMTEntity(self.context, assigned, element=element)
        return obj

    def multicol(self, *args):
        obj = AMTMultiCol(self.context, *args)
        self.addChild(obj)
        return obj

    def get(self, spaceManager):
        assert len(self.children) == 1

        # if self.context.sameRowsSpaces and len(spaces) == 1:
        #     spaces = [spaces[0]] * len(list(allAssignables.values())[0])

        return list(self.children)[0].get(spaceManager)

    # Abstract
    def abstractEntity(self, entity, abstractNewElements=True):
        if id(entity) in self.context.matching:
            return self.entity(element=self.context.matching[id(entity)])
        if abstractNewElements:
            obj = self.entity()
        else:
            obj = self.entity(entity)
        self.context.matching[id(entity)] = obj.element
        obj.element.assignable(entity)
        return obj

    def abstractProperty(self, property, kind=SpaceKind.BASIC, abstractNewElements=True, appendToChildren=False):
        entity = self.abstractEntity(
            property.entity, abstractNewElements=abstractNewElements)
        if id(property) in self.context.matching:
            return entity.property(element=self.context.matching[id(property)], kind=kind)
        obj = entity.property(propertyName=property.name, kind=kind)
        self.context.matching[id(property)] = obj.element
        if appendToChildren:
            self.addChild(obj)
        return obj

    def abstractSpace(self, space, abstractNewElements=True, appendToChildren=False):
        if not space.aggregation:
            if space.spaces[0] == space:
                property = space.boundProperty
                kind = space.kind
                # entity = self.abstractEntity(property.entity, abstractNewElements=abstractNewElements)
                expr = self.abstractProperty(
                    property, kind=kind, abstractNewElements=abstractNewElements)
            else:
                pass
        else:
            expr = self.multicol(
                *[self.abstractSpace(s, abstractNewElements=abstractNewElements) for s in space.flatCols])
        if appendToChildren:
            self.addChild(expr)
        return expr


class AMTExpr(AMTBase):
    def __init__(self, context, item):
        super().__init__(context)
        self.addChild(item)


class AMTMultiCol(AMTBase):
    def __init__(self, context, *args):
        super().__init__(context)
        self.addChildren(args)

    def get(self, spaceManager):
        spaces = []
        for child in self.children:
            s = child.get(spaceManager)
            s.append(s)
        space = spaceManager.multiColSpace(spaces)
        return space

    def toStr(self):
        return '[{}]'.format('+'.join([item.toStr() for item in self.children]))


class AMTElementLinked(AMTBase):
    def __init__(self, context, assigned=None, parent=None, element=None):
        if not element:
            element = AMTElement(context, self.__class__, assigned=assigned)
        super().__init__(context, parent=parent, element=element)

    @property
    def abstract(self):
        return self.element.abstract

    def assignable(self, obj):
        self.element.assignable(obj)

    def unassignable(self, obj):
        self.element.unassignable(obj)

    def toStr(self):
        return self.fullname()


class AMTEntity(AMTElementLinked):
    NAME = 'Entity'

    def __init__(self, context, entity=None, element=None):
        super().__init__(context, element=element, assigned=entity)

    def property(self, propertyName=None, element=None, kind=SpaceKind.BASIC):
        return AMTProperty(self.context, self, propertyName=propertyName, element=element, kind=kind)

    def get(self, spaceManager):
        return self.element.assignables

    @staticmethod
    def _namePattern(obj, index):
        return '({})'.format(chr(ord('A') + index))

    def _fullnamePattern(self, name):
        return name


class AMTProperty(AMTElementLinked):
    NAME = 'Property'

    def __init__(self, context, entity, propertyName=None, element=None, kind=SpaceKind.BASIC):
        super().__init__(context, element=element, assigned=propertyName)
        self.kind = kind
        self.addChild(entity)

    def toStr(self):
        return self.fullname()

    @property
    def entity(self):
        return list(self.children)[0]

    def get(self, spaceManager):
        if len(self.element.assignables) == 0:
            raise Exception(
                '{} has no possible assignable values!'.format(self.element))

        if self.context.sameRowsSpaces:
            allAssignables = self.context.assignableElements()
            if len(list(allAssignables.keys())) != 1:
                raise Exception(
                    'Only one assignable may vary with \'sameRowsSpaces\' enabled')

        properties = []
        for propertyName in list(self.element.assignables):
            for entity in self.entity.get(spaceManager):
                prop = entity.property(propertyName)
                if prop is None:
                    raise Exception(
                        '{} has no property {}!'.format(entity, propertyName))
                properties.append(prop)
        spaces = [spaceManager.spaceSearch(
            property=property, kind=self.kind) for property in properties]

        if self.context.sameRowsSpaces and len(spaces) == 1:
            spaces = [spaces[0]] * len(list(allAssignables.values())[0])

        return spaceManager.multiRowSpace(spaces)

    @staticmethod
    def _namePattern(obj, index):
        return '({})'.format(chr(ord('a') + index))

    def _fullnamePattern(self, name):
        return '{}.{}'.format(self.entity.name(), name)


class AMTElement(object):
    def __init__(self, context, cls, assigned=None):
        if assigned:
            name = repr(assigned)
        self.context = context
        self.cls = cls
        self.assignables = set()
        self.unassignables = set()
        self.assigned = assigned
        if self.assigned:
            self.assignable(self.assigned)

    def name(self):
        if id(self) in self.context.names:
            return self.context.names[id(self)]
        if self.assigned:
            name = str(self.assigned)
            self.context.names[id(self)] = name
            # self.context.fullnames[id(self)] = self.cls._fullnamePattern(self, self._name)
            return name

        name = '__base__'
        i = 0
        while name in self.context.names.values():
            name = self.cls._namePattern(self, i)
            # fullname = self.cls._fullnamePattern(self, name)
            i += 1

        self.context.names[id(self)] = name
        # self.context.fullnames[id(self)] = fullname
        return name

    @property
    def abstract(self):
        return self.assigned is None

    def assignable(self, obj):
        self.assignables.add(obj)
        print('hello')
        if obj in self.unassignables:
            self.unassignables.remove(obj)
        if len(self.assignables) > 1:
            self.assigned = None

    def unassignable(self, obj):
        self.unassignables.add(obj)
        if obj in self.assignables:
            self.assignables.remove(obj)

    def __repr__(self):
        return '{} {}'.format(self.cls.NAME, self.name())