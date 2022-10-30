from .internal_engine import InternalEngine, GraphicalEngine, DrawOptions

import numpy as np

from PIL import Image
from scipy.spatial.distance import euclidean

# PyMunk
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

# PyGame
import pygame
import pygame.locals as pglocals
from pygame.color import THECOLORS

# Panda 3D
from panda3d.core import Point3, Vec3, ClockObject, loadPrcFileData, AntialiasAttrib, PointLight, AmbientLight, DirectionalLight, Spotlight, Filename, getModelPath


class PymunkEngine(InternalEngine):
    ENGINE = 'Pymunk'

    # def __init__(self, environment, options={}):
    #     super().__init__(environment, options)

    def setupGraphicalEngines(self):
        self.registerGraphicalEngine(PyGameGE, True)
        self.registerGraphicalEngine(Panda3DGE)

    def initPhysics(self):
        # Physics stuff
        self.physics = pymunk.Space()
        self.physics.gravity = (0.0, 0.0)
        self.physics.damping = 0.1

    
    # def reset(self):
    #     for obj in self.agents + self.objs:
    #         if isinstance(obj, Physical):
    #             obj.body.position = obj.coords_init
    #             obj.body.angle = 0
    #             obj.body.angular_velocity = 0
    #             obj.body.velocity = Vec2d(0.0, 0.0)
    #     self.setupIteration()

    def _updatePhysics(self, dt):
        self.physics.step(dt)

    def checkSegmentOccupancy(self, x1, y1, x2, y2, width=0, exceptions=[]):
        exceptionShapes = [entity.shape for entity in exceptions]
        pos = Vec2d(x1, y1)
        dest = Vec2d(x2, y2)
        hits = self.physics.segment_query(
            pos, dest, width, pymunk.ShapeFilter())
        filteredHits = [
            info for info in hits if info.shape not in exceptionShapes]
        return len(filteredHits) > 0

    def checkCaseOccupancy(self, x, y, w, h, exceptions=[]):
        exceptionShapes = [entity.shape for entity in exceptions]

        body = pymunk.Body(1., 1.)
        body.position = x + w / 2, y + h / 2
        shape = pymunk.Poly(
            body, [(-w/2, -h/2), (-w/2, h/2), (w/2, h/2), (w/2, -h/2), (-w/2, -h/2)])

        hits = self.physics.shape_query(shape)
        filteredHits = [
            info for info in hits if info.shape not in exceptionShapes]
        return len(filteredHits) > 0

    def checkGridOccupancy(self, x1, y1, x2, y2, caseW, caseH, exceptions=[]):
        numX = (x2 - x1) // caseW
        numY = (y2 - y1) // caseH
        grid = np.zeros((numY, numX))

        for i in range(numY):
            for j in range(numX):
                grid[i, j] = self.checkCaseOccupancy(
                    x1 + caseW*j, y1 + caseH*i, caseW, caseH, exceptions=exceptions)

        return grid

    def customStep(self, duration=0.):
        if self.displayThisFrame:
            for g in self.graphicalEngines:
                g.customStep(duration)


class PyGameGE(GraphicalEngine):
    def _initGraphics(self):
        pygame.init()
        self._updateGraphics(False)
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.font = pygame.font.SysFont("monospace", 13)

    def _updateGraphics(self, oldgui):
        if self.gui:
            self.screen = pygame.display.set_mode(self.DISPLAY_SIZE)
        else:
            if oldgui:
                pygame.display.quit()
            self.screen = pygame.Surface(self.DISPLAY_SIZE)

    def _image(self):
        self.engine.updatePhysics()
        self.drawScreen()

        pil_string_image = pygame.image.tostring(self.screen, "RGBA", False)
        pil_image = Image.frombytes("RGBA", self.DISPLAY_SIZE, pil_string_image)

        return pil_image

    def flipScreen(self, delay=False):
        if self.gui:
            pygame.display.flip()
            pygame.event.get()
            if delay:
                self.clock.tick(50*(self.engine.speedupFrames if self.engine.speedupFrames is not None else 1000000))
                pygame.display.set_caption(f"{self.engine.windowTitle()} | GUI fps: {self.clock.get_fps():.0f}")

    def _hide(self):
        pygame.display.set_caption(
            f"{self.engine.windowTitle()} | GUI Stopped (simulation may continue in background)")

    def _show(self):
        pygame.display.set_caption(
            f"{self.engine.windowTitle()} | GUI Ready (simulation on pause)")

    def _drawScreen(self, duration=0.):
        self.screen.fill(THECOLORS["white"])  # Clear screen
        #self.physics.debug_draw(self.draw_options) 
        drawOptions = PMDrawOptions(self, False)
        self._drawSelfScene(self.screen, drawOptions)
        for entity in self.environment.world.cascadingChildren():
            entity.draw(self.screen, drawOptions)
        if duration:
            label_acting = self.font.render("Acting", 1, (255, 0, 0))
            self.screen.blit(label_acting, (24, 24))
        else:
            pass
            # label_mode = self.font.render("Current outcome space: / [J] to change", 1,
            #                               (0, 0, 0))
            # self.screen.blit(label_mode, (24, 24))
    
    def customStep(self, duration=0.):
        if self.gui and duration is None:
            for event in pygame.event.get():
                if event.type == pglocals.QUIT:
                    self.running = False
                elif event.type == pglocals.KEYDOWN and event.key == pglocals.K_ESCAPE:
                    self.running = False
                elif event.type == pglocals.KEYDOWN and event.key == pglocals.K_j:
                    # self.agents[0].execute_action((1.0, 0.0))
                    # change mode
                    if self.current_space == self.env.dataset.outcomeSpaces[-1]:
                        self.current_space = self.env.dataset.outcomeSpaces[0]
                    else:
                        self.current_space = self.env.dataset.outcomeSpaces[self.env.dataset.outcomeSpaces.index(
                            self.current_space) + 1]
                elif event.type == pygame.MOUSEBUTTONUP:
                    # convert to local agent coordinates
                    pos = self.propertyItem(self.current_space.property).convert_to_feature(
                        pymunk.pygame_util.from_pygame(event.pos, self.screen))
                    self.experiment.exploit(
                        pos, self.env.dataset.outcomeSpaces.index(self.current_space))

        # Draw stuff
        if self.gui or self._record:
            self.drawScreen(duration)


class Panda3DGE(GraphicalEngine):
    def pandaResourcesPath(self, filepath=''):
        return Filename.fromOsSpecific(str(self.engine.environment.resourcesPath(filepath).absolute())).getFullpath()

    def _initGraphics(self):
        self._updateGraphics(False)

        from direct.showbase import ShowBase
        self.base = ShowBase.ShowBase()

        getModelPath().appendDirectory(self.pandaResourcesPath())

        # import simplepbr
        # pipeline = simplepbr.init()
        # pipeline.enable_shadows = True

        # if self.gui:
        self.base.cam.setPos(50, 50, 50)
        self.base.cam.lookAt(0, 0, 0)
        # self.base.setFrameRateMeter(True)

        # self.base.render.setAntialias(AntialiasAttrib.MMultisample)
        # self.base.render.setAntialias(AntialiasAttrib.MMultisample, 4)
        self.base.render.setAntialias(AntialiasAttrib.MAuto)

        # teapot = loader.loadModel('teapot')
        # teapot.reparentTo(self.base.render)
        # teapot.setPos(100, 100, 50)
        # teapot.setScale(20.)
        # teapotMovement = teapot.hprInterval(50, Point3(0, 360, 360))
        # teapotMovement.loop()

        self.sun = self.base.render.attachNewNode(Spotlight("Spot"))
        self.sun.node().setScene(self.base.render)
        self.sun.node().setShadowCaster(True, 4096, 4096)
        # self.sun.node().showFrustum()
        self.sun.node().getLens().setFov(60)
        self.sun.node().getLens().setNearFar(10, 2000)
        self.sun.node().setColor((1., 1., 1., 1))
        self.base.render.setLight(self.sun)

        # self.sun = DirectionalLight('sun')
        # self.sun.setColor((0.8, 0.8, 0.5, 1))
        # self.sun = PointLight('sun')
        # self.sun.setColor((0.9, 0.9, 0.9, 1))
        # self.sun.attenuation = (1, 0, 0)
        # self.sun.setShadowCaster(True, 64, 64)

        # self.sunp = self.base.render.attachNewNode(self.sun)
        # self.sunp.setHpr(50, -60, 0)
        self.sun.setPos(500, 550, 1200)
        self.sun.lookAt(300, 300, 0)

        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1))
        alnp = self.base.render.attachNewNode(alight)
        self.base.render.setLight(alnp)

        from panda3d.core import RenderState, CullFaceAttrib, ColorWriteAttrib

        self.sun.node().setInitialState(RenderState.make(
            CullFaceAttrib.makeReverse(), ColorWriteAttrib.make(ColorWriteAttrib.COff)))

        self.base.render.setShaderAuto()

    def _updateGraphics(self, oldgui):
        self.gui = True
        if self.gui:
            loadPrcFileData("", "window-title Hello")
            loadPrcFileData("", "win-size 1800 600")
            loadPrcFileData("", "sync-video 0")
        else:
            if oldgui:
                pass
            loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "framebuffer-multisample 1")
        loadPrcFileData("", "multisamples 2")

    def _image(self):
        self.engine.updatePhysics()
        self.drawScreen()

        self.base.graphicsEngine.renderFrame()
        self.base.graphicsEngine.renderFrame()

        FILENAME = 'tmp.png'
        self.base.screenshot(FILENAME, defaultFilename=False)
        imageWithAlpha = Image.open(FILENAME).convert('RGBA')

        image = Image.new("RGB", imageWithAlpha.size, (128, 128, 128))
        image.paste(imageWithAlpha, mask=imageWithAlpha.split()[3])

        return image

    def flipScreen(self, delay=False):
        if self.gui:
            self.base.graphicsEngine.renderFrame()
            if delay:
                pass

    def _hide(self):
        pass

    def _show(self):
        pass

    def _drawScreen(self, duration=0.):
        drawOptions = PMDrawOptions(self, True)
        self._drawSelfScene(self.base, drawOptions)
        for entity in self.environment.world.cascadingChildren():
            entity.draw(self.base, drawOptions)

    def customStep(self, duration=0.):
        # if self.gui and duration is None:
        #     for event in pygame.event.get():
        #         if event.type == pglocals.QUIT:
        #             self.running = False
        #         elif event.type == pglocals.KEYDOWN and event.key == pglocals.K_ESCAPE:
        #             self.running = False
        #         elif event.type == pglocals.KEYDOWN and event.key == pglocals.K_j:
        #             # self.agents[0].execute_action((1.0, 0.0))
        #             # change mode
        #             if self.current_space == self.env.dataset.outcomeSpaces[-1]:
        #                 self.current_space = self.env.dataset.outcomeSpaces[0]
        #             else:
        #                 self.current_space = self.env.dataset.outcomeSpaces[self.env.dataset.outcomeSpaces.index(
        #                     self.current_space) + 1]
        #         elif event.type == pygame.MOUSEBUTTONUP:
        #             # convert to local agent coordinates
        #             pos = self.propertyItem(self.current_space.property).convert_to_feature(
        #                 pymunk.pygame_util.from_pygame(event.pos, self.screen))
        #             self.experiment.exploit(
        #                 pos, self.env.dataset.outcomeSpaces.index(self.current_space))

        # Draw stuff
        if self.gui or self._record:
            self.drawScreen(duration)


class PMDrawOptions(DrawOptions):
    def __init__(self, graphicalEngine, panda):
        super().__init__(graphicalEngine)
        self.panda = panda
        self.pygame = not panda
