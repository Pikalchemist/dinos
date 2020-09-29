from .internal_engine import InternalEngine

import numpy as np

import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

from PIL import Image
from scipy.spatial.distance import euclidean


class PymunkEngine(InternalEngine):
    ENGINE = 'Pymunk'

    # def __init__(self, environment, options={}):
    #     super().__init__(environment, options)

    def initPhysics(self):
        # Physics stuff
        self.physics = pymunk.Space()
        self.physics.gravity = (0.0, 0.0)
        self.physics.damping = 0.1

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

    # def reset(self):
    #     for obj in self.agents + self.objs:
    #         if isinstance(obj, Physical):
    #             obj.body.position = obj.coords_init
    #             obj.body.angle = 0
    #             obj.body.angular_velocity = 0
    #             obj.body.velocity = Vec2d(0.0, 0.0)
    #     self.setupIteration()

    def _image(self):
        self.updatePhysics()
        self.drawScreen()

        pil_string_image = pygame.image.tostring(self.screen, "RGBA", False)
        pil_image = Image.frombytes("RGBA", (600, 600), pil_string_image)

        return pil_image

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

    def flipScreen(self, delay=False):
        if self.gui:
            pygame.display.flip()
            pygame.event.get()
            if delay:
                self.clock.tick(
                    50*(self.speedupFrames if self.speedupFrames is not None else 1000000))
                pygame.display.set_caption(
                    f"{self.windowTitle()} | GUI fps: {self.clock.get_fps():.0f}")

    def _hide(self):
        pygame.display.set_caption(
            f"{self.windowTitle()} | GUI Stopped (simulation may continue in background)")

    def _show(self):
        pygame.display.set_caption(
            f"{self.windowTitle()} | GUI Ready (simulation on pause)")

    def _drawScreen(self, duration=0.):
        self.screen.fill(THECOLORS["white"])  # Clear screen
        self._drawSelfScene()
        #self.physics.debug_draw(self.draw_options)
        for entity in self.world.cascadingChildren():
            entity.draw(self.screen)
        if duration:
            label_acting = self.font.render("Acting", 1, (255, 0, 0))
            self.screen.blit(label_acting, (24, 24))
        else:
            pass
            # label_mode = self.font.render("Current outcome space: / [J] to change", 1,
            #                               (0, 0, 0))
            # self.screen.blit(label_mode, (24, 24))

    def customStep(self, duration=0.):
        if not self.displayThisFrame:
            return

        if self.gui and duration is None:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_j:
                    # self.agents[0].execute_action((1.0, 0.0))
                    # change mode
                    if self.current_space == self.env.dataset.outcomeSpaces[-1]:
                        self.current_space = self.env.dataset.outcomeSpaces[0]
                    else:
                        self.current_space = self.env.dataset.outcomeSpaces[self.env.dataset.outcomeSpaces.index(
                            self.current_space) + 1]
                elif event.type == pygame.MOUSEBUTTONUP:
                    # convert to local agent coordinates
                    pos = self.property(self.current_space.property).convert_to_feature(
                        pymunk.pygame_util.from_pygame(event.pos, self.screen))
                    self.experiment.exploit(
                        pos, self.env.dataset.outcomeSpaces.index(self.current_space))

        # Draw stuff
        if self.gui or self._record:
            self.drawScreen(duration)
