from exlab.lab.lab import Lab
from exlab.lab.counter import EpisodeAbsoluteIterationCounter
from exlab.utils.path import basepath


basedir = basepath(__file__)

lab = Lab(basedir, defaults='defaults.yml', help='Simple example')

lab.parameter('infos.dataset').import_key('datasets', 'data.yml')
lab.parameter('learner').import_file(['learners', 'learners2'])

lab.filter('learners/*').parameter('dataset').import_key('datasets', 'data.yml')

lab.load(counter=EpisodeAbsoluteIterationCounter)

def run(exp):
    print(exp.config['learner'])

lab.run(run)
