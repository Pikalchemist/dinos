'''
    File name: environment.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

from PIL import Image


class Engine(object):
    PHYSICAL = False

    DISPLAY_SIZE = (600, 600)

    IMAGES_MAX = 400
    IMAGE_WIDTH = 300
    VIDEO_FPS = 3
    VIDEO_PERIOD = 1. / float(VIDEO_FPS)
    FREQ_SIM = 60.0
    PHYSICS_STEPS = 5

    def __init__(self, environment, options={}):
        self.environment = environment

        # Time variables
        self.iteration = 0
        self.time = 0.

        # Options
        self.options = options

        # Simu speed
        self.speedupFrames = options.get('speedup-frames', 20.)
        self.skippedFrames = options.get('skipped-frame', 10)
        self.jumpCurrentFrame = 0
        self.displayThisFrame = True

        # Image / Video
        self._record = False
        self._recordOnce = False
        self._images = []

        # GUI
        self._gui = False
        self.graphical = False
        self.graphicsLoaded = False
        self.gui = options.get("gui", False)

        self.initPhysics()
    
    @property
    def scene(self):
        return self.environment.scene
    
    @property
    def world(self):
        return self.environment.world

    # Graphics
    def show(self):
        self.displayGui(True)
    
    def hide(self):
        self.displayGui(False)

    def displayGui(self, gui=True):
        if self.gui == gui and self.graphical:
            return

        oldgui = self.gui
        self.gui = gui
        if self.gui:
            self.useGraphics(True)
            self._updateGraphics(oldgui)
            self._show()
            self.display()
        else:
            self._updateGraphics(oldgui)
            self._hide()

    def useGraphics(self, graphical=True, gui=None):
        if self.graphical != graphical:
            self.graphical = graphical
            if self.graphical and not self.graphicsLoaded:
                self.graphicsLoaded = True
                self._initGraphics()

        if gui is not None:
            self.displayGui(gui)

    def _initGraphics(self):
        pass

    def _updateGraphics(self, oldgui):
        pass

    def _hide(self):
        pass

    def _show(self):
        pass

    def _image(self):
        pass

    def _drawScreen(self):
        pass

    def flipScreen(self):
        pass

    def windowTitle(self):
        return f'Dino {self.environment.name} - {self.scene.name}'

    # Image / Video
    def display(self):
        return self.image()

    def render(self):
        return self.display()

    def image(self):
        self.useGraphics()
        img = self._image()
        self.flipScreen()
        hsize = int((float(img.size[1])*float(self.IMAGE_WIDTH / img.size[0])))
        img = img.resize((self.IMAGE_WIDTH, hsize), Image.ANTIALIAS)
        return img

    def video(self, format='MP4V', fps=10):
        image_array = [utils.video.image_PIL2np(
            image) for image in self._images]
        utils.video.display_videojs(image_array[::])

    def _toVideo(self):
        if len(self._images) > self.IMAGES_MAX:
            self._images.pop(0)
        self._images.append(self.image())

    def recordOnce(self):
        self._images = []
        self._record = True
        self._recordOnce = True
        self._lastVideoImage = 0.

    def record(self, record=True):
        if record:
            self._images = []
        self._record = record
        self._recordOnce = False
        self._lastVideoImage = 0.

    def drawScreen(self, duration=0.):
        self._drawScreen(duration)

    def _drawSelfScene(self):
        self._draw()
        self.scene._draw()

    def _draw(self):
        pass

    # Physics
    def initPhysics(self):
        pass

    def updatePhysics(self, dt):
        pass

    # Frames
    def checkJumpFrame(self):
        self.displayThisFrame = True
        if self.skippedFrames == 0:
            return True
        self.jumpCurrentFrame += 1
        if self.jumpCurrentFrame > self.skippedFrames:
            self.jumpCurrentFrame = 0
            return True
        self.displayThisFrame = False
        return False

    def simu(self):
        graphical = (
            'displayed' if self.gui else 'hidden') if self.graphical else 'graphical disabled'
        speed = 'x{}'.format(
            self.speedupFrames) if self.speedupFrames is not None else 'max'
        jump = 'displays all frames'
        if self.skippedFrames > 0:
            jump = 'displays 1 frame every {} frames'.format(
                self.skippedFrames)
        return 'Environment {}\nGUI: {} - fps {} - {}\n'.format(self.__class__.__name__, graphical, speed, jump)

    # Main loop
    def run(self, duration=None):
        elapsedTime = 0
        self.running = True
        if not duration:
            duration = self.environment.timestep
        while self.running:
            self.checkJumpFrame()
            self.customStep(duration)

            dt = 1.0 / self.FREQ_SIM

            # Update physics
            if self.PHYSICAL:
                subdt = dt / float(self.PHYSICS_STEPS)
                for _ in range(self.PHYSICS_STEPS):
                    self.updatePhysics(subdt)

                    # Update
                    self.environment.world.update(subdt)

            # Flip screen
            if self.displayThisFrame:
                self.flipScreen(delay=True)
            if self._record and self.time >= self._lastVideoImage + self.VIDEO_PERIOD:
                self._lastVideoImage = self.time
                self._toVideo()
            elapsedTime += dt
            if duration and elapsedTime >= duration:
                self.running = False
        self.running = True

        if self._recordOnce:
            self._record = False
            self._recordOnce = False
