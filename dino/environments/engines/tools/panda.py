from panda3d.core import Point3, Vec3, Vec4, Point2, LPoint2i, Geom, GeomNode, GeomTriangles, GeomVertexFormat, GeomVertexData, GeomVertexWriter

import numpy as np


class PandaElement(object):
    def __init__(self, pt):
        self.pt = pt
        self.pt.elements.append(self)

        self.written = False
        self.color = Vec4(self.pt.color)
        self.normal = None
        self.normalFirst2 = None

    def alpha(self, alpha):
        self.color.setZ(alpha)
    
    def findNormal(self, a, b, c):
        ab = b - a
        ac = c - a
        # print('===')
        # print(ab)
        # print(ac)
        return ab.cross(ac).normalized()
    
    def writeData(self, points, normals, addPrimitives=True):
        self.written = True
        if not isinstance(normals, (list, tuple)):
            normals = [normals] * len(points)

        index = self.pt.vertex.getWriteRow()
        for point, normal in zip(points, normals):
            self.pt.vertex.addData3f(point)
            self.pt.vnormal.addData3f(normal)
            self.pt.vcolor.addData4f(self.color)
        
        if addPrimitives:
            self.pt.primitive.addVertices(index, index + 1, index + 2)
            if len(points) >= 4:
                self.pt.primitive.addVertices(index, index + 2, index + 3)

        return index

class PandaElementTriangle(PandaElement):
    def setup(self, a, b, c, reverse=False):
        if not reverse:
            self.a, self.b, self.c = a, b, c
        else:
            self.a, self.b, self.c = c, b, a
        return self
    
    def write(self):
        points = [self.a, self.b, self.c]
        self.normal = self.findNormal(self.a, self.b, self.c)

        self.writeData(points, self.normal)

class PandaElementPlane(PandaElement):
    def setup(self, a, b, c, d, reverse=False):
        if not reverse:
            self.a, self.b, self.c, self.d = a, b, c, d
        else:
            self.a, self.b, self.c, self.d = d, c, b, a
        return self
    
    def write(self):
        points = [self.a, self.b, self.c, self.d]

        self.normal = self.findNormal(self.a, self.b, self.c)
        # print(self.normal)
        normals = [self.normal] * 4
        if self.normalFirst2:
            normals[0], normals[1] = [self.normalFirst2, self.normalFirst2]

        self.writeData(points, normals)


class PandaTools(object):
    def __init__(self, autoStartPrimitive=True):
        if autoStartPrimitive:
            self.startPrimitive()
        self.elements = []
        self.color = Vec4(1., 1., 1., 1.)

    def plane(self, a, b, c, d, **options):
        return [PandaElementPlane(self).setup(a, b, c, d)]
    
    def horizontalPlane(self, a, c, **options):
        b = Point3(c.x, a.y, a.z)
        d = Point3(a.x, c.y, c.z)
        return [PandaElementPlane(self).setup(a, b, c, d)]
    
    def verticalPlane(self, a, c, **options):
        b = Point3(c.x, c.y, a.z)
        d = Point3(a.x, a.y, c.z)
        return [PandaElementPlane(self).setup(a, b, c, d)]
    
    def cube(self, a, c, z, **options):
        b = Point3(c.x, a.y, a.z)
        d = Point3(a.x, c.y, c.z)
        return self.quadri(a, b, c, d, z, **options)

    def quadri(self, a, b, c, d, z, **options):
        def z0(p):
            return Point3(p.x, p.y, z)

        return [PandaElementPlane(self).setup(a, b, c, d),
                PandaElementPlane(self).setup(z0(a), z0(b), z0(c), z0(d), reverse=True),
                self.verticalPlane(c, z0(b))[0],
                self.verticalPlane(d, z0(c))[0],
                self.verticalPlane(a, z0(d))[0],
                self.verticalPlane(b, z0(a))[0]]
    
    def cylinder(self, radius, height, origin=Point3.zero(), steps=50):
        elements = []
        upCenter = origin + Vec3(0, 0, height)
        for i in range(steps):
            a = origin + Vec3(np.cos(np.math.pi * 2 / steps * i) * radius,
                              np.sin(np.math.pi * 2 / steps * i) * radius, 0)
            c = origin + Vec3(np.cos(np.math.pi * 2 / steps * (i + 1)) * radius,
                              np.sin(np.math.pi * 2 / steps * (i + 1)) * radius, height)
            elements.append(self.verticalPlane(a, c)[0])
            elements.append(PandaElementTriangle(self).setup(upCenter, Point3(a.x, a.y, a.z + height), c))
            elements.append(PandaElementTriangle(self).setup(origin, Point3(c.x, c.y, c.z - height), a))
        return elements

    def startPrimitive(self):
        self.vdata = GeomVertexData('chunk-vertex', GeomVertexFormat.getV3n3c4(), Geom.UHStatic)

        self.vertex = GeomVertexWriter(self.vdata, 'vertex')
        self.vnormal = GeomVertexWriter(self.vdata, 'normal')
        self.vcolor = GeomVertexWriter(self.vdata, 'color')

        self.primitive = GeomTriangles(Geom.UHStatic)
    
    def endPrimitive(self):
        self.primitive.closePrimitive()
    
    def end(self):
        for element in self.elements:
            if not element.written:
                element.write()
        self.endPrimitive()
        return self.createNode()
    
    def createNode(self):
        self.node = GeomNode('chunk')

        geom = Geom(self.vdata)
        geom.addPrimitive(self.primitive)
        self.node.addGeom(geom)
    
        return self.node
