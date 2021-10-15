'''
    File name: environment.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

from PIL import Image
from dinos import utils


class Engine(object):
    PHYSICAL = False
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
        self.skippedFrames = options.get('skipped-frame', 50)
        self.jumpCurrentFrame = 0
        self.displayThisFrame = True

        self.graphicalEngine = None
        self.graphicalEngines = []
        self.setupGraphicalEngines()

        self.initPhysics()
    
    def setupGraphicalEngines(self):
        pass
    
    def registerGraphicalEngine(self, cls, default=False):
        g = cls(self, self.options)
        if self.graphicalEngine is None or default:
            self.graphicalEngine = g
        if g not in self.graphicalEngines:
            self.graphicalEngines.append(g)
    
    @property
    def scene(self):
        return self.environment.scene
    
    @property
    def world(self):
        return self.environment.world
    
    def windowTitle(self):
        return f'Dinos {self.environment.name} - {self.scene.name} - {self.environment.iteration}'
    
    # Graphics
    def show(self):
        self.graphicalEngine.show()

    def hide(self):
        self.graphicalEngine.hide()

    # Image / Video
    def display(self):
        return self.graphicalEngine.display()

    def render(self):
        return self.display()

    def recordOnce(self):
        return self.graphicalEngine.recordOnce()

    def record(self, record=True):
        return self.graphicalEngine.record(record)
    
    def drawScreen(self, duration=0.):
        self.graphicalEngine.drawScreen(duration)

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
            'displayed' if self.graphicalEngine.gui else 'hidden') if self.graphicalEngine.graphical else 'graphical disabled'
        speed = f'x{self.speedupFrames}' if self.speedupFrames is not None else 'max'
        jump = 'displays all frames'
        if self.skippedFrames > 0:
            jump = f'displays 1 frame every {self.skippedFrames} frames'
        return f'Environment {self.__class__.__name__}\nGUI: {graphical} - fps {speed} - {jump}\n'

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
            for g in self.graphicalEngines:
                if self.displayThisFrame:
                    g.flipScreen(delay=True)
                g.checkRecording(self.time)
            elapsedTime += dt
            if duration and elapsedTime >= duration:
                self.running = False
        self.running = True
        for g in self.graphicalEngines:
            g.checkRecordingOnce()
    
    def customStep(self, duration=0.):
        pass


class GraphicalEngine(object):
    DISPLAY_SIZE = (600, 600)
    IMAGES_MAX = 400
    IMAGE_WIDTH = 300
    VIDEO_FPS = 3
    VIDEO_PERIOD = 1. / float(VIDEO_FPS)

    def __init__(self, engine, options):
        self.engine = engine

        # Image / Video
        self._record = False
        self._recordOnce = False
        self._images = []

        # GUI
        self._gui = False
        self.graphical = False
        self.graphicsLoaded = False
        self.gui = options.get("gui", False)
    
    def __repr__(self):
        return f'GraphicalEngine {self.__class__.__name__}'
    
    @property
    def environment(self):
        return self.engine.environment
    
    def show(self):
        self.displayGui(True)

    def hide(self):
        self.displayGui(False)
    
    def display(self):
        return self.image()

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

    def _drawScreen(self, duration=0.):
        pass

    def _drawSelfScene(self, base, drawOptions):
        self._draw(base, drawOptions)
        self.environment.scene._draw(base, drawOptions)

    def _draw(self, base, drawOptions):
        pass

    def drawScreen(self, duration=0.):
        self._drawScreen(duration)

    def flipScreen(self):
        pass

    def customStep(self, duration=0.):
        pass

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
    
    def checkRecording(self, time):
        if self._record and time >= self._lastVideoImage + self.VIDEO_PERIOD:
            self._lastVideoImage = time
            self._toVideo()
    
    def checkRecordingOnce(self):
        if self._recordOnce:
            self._record = False
            self._recordOnce = False
    
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


class DrawOptions(object):
    def __init__(self, graphicalEngine):
        self.graphicalEngine = graphicalEngine
    
    @property
    def environment(self):
        return self.graphicalEngine.engine.environment
