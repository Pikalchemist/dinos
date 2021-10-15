# from .internal_engine import InternalEngine

# from panda3d.core import Vec3
# from panda3d.bullet import BulletWorld
# from panda3d.bullet import BulletPlaneShape
# from panda3d.bullet import BulletRigidBodyNode
# from panda3d.bullet import BulletBoxShape
# from pandac.PandaModules import ClockObject
# from pandac.PandaModules import loadPrcFileData


# class PandaEnvironment(InternalEngine):
#     ENGINE = 'Panda3D'

#     def __init__(self, scene=None, options={}):
#         super().__init__(scene, options)

#         self.jump_frame = self.speedup_frames
#         self.jump_physics_frame = 0

#     def initPhysics(self):
#         # Physics stuff
#         # World
#         self.space = BulletWorld()
#         self.space.setGravity(Vec3(0, 0, -9.81))

#         # Plane
#         shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
#         node = BulletRigidBodyNode('Ground')
#         node.addShape(shape)
#         np = render.attachNewNode(node)
#         np.setPos(0, 0, -2)
#         self.space.attachRigidBody(node)

#     def setGui(self, gui):
#         if self.gui == gui:
#             return
#         self.gui = gui
#         if self.gui:
#             loadPrcFileData("", "window-title Your Title")
#             loadPrcFileData("", "win-size 800 600")
#             loadPrcFileData("", "sync-video 0")
#         else:
#             loadPrcFileData("", "window-type none")

#         from direct.showbase import ShowBase
#         base = ShowBase.ShowBase()

#         if self.gui:
#             base.cam.setPos(0, -10, 0)
#             base.cam.lookAt(0, 0, 0)
#             base.setFrameRateMeter(True)

#     def reset(self):
#         for obj in self.agents + self.objs:
#             pass  # TODO

#     def _updatePhysics(self, dt):
#         self.jump_physics_frame += 1
#         if self.jump_physics_frame >= 5:
#             self.space.doPhysics(0.001)
#             self.jump_physics_frame = 0

#     def flipScreen(self):
#         self.jump_frame += 1.
#         if self.jump_frame >= self.speedup_frames:
#             taskMgr.step()
#             self.jump_frame = 0.

#     def customStep(self, duration):
#         '''if self.gui and duration is None:
#             for event in pygame.event.get():
#                 if event.type == QUIT:
#                     self.running = False
#                 elif event.type == KEYDOWN and event.key == K_ESCAPE:
#                     self.running = False
#                 elif event.type == KEYDOWN and event.key == K_j:
#                     # self.agents[0].execute_action((1.0, 0.0))
#                     # change mode
#                     if self.current_space == self.env.dataset.outcomeSpaces[-1]:
#                         self.current_space = self.env.dataset.outcomeSpaces[0]
#                     else:
#                         self.current_space = self.env.dataset.outcomeSpaces[self.env.dataset.outcomeSpaces.index(self.current_space) + 1]
#                 elif event.type == pygame.MOUSEBUTTONUP:
#                     # convert to local agent coordinates
#                     pos = self.propertyItem(self.current_space.property).convert_to_feature(pymunk.pygame_util.from_pygame(event.pos, self.screen))
#                     self.experiment.exploit(pos, self.env.dataset.outcomeSpaces.index(self.current_space))'''

#         # Draw stuff
#         '''if self.gui:
#             self.screen.fill(THECOLORS["white"])  # Clear screen
#             self.space.debug_draw(self.draw_options)
#             if duration:
#                 label_acting = self.font.render("Acting", 1, (255, 0, 0))
#                 self.screen.blit(label_acting, (24, 24))
#             else:
#                 label_mode = self.font.render("Current outcome space: " + self.current_space.name + " [J] to change", 1,
#                                               (0, 0, 0))
#                 self.screen.blit(label_mode, (24, 24))'''
