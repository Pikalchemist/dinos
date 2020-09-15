'''
    File name: dataset.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from dino.data.space import SpaceKind
from dino.agents.tools.models.model import Model

from .dataset import Dataset


class ModelDataset(Dataset):
    """The dataset used to record actions, outcomes and procedures."""

    def __init__(self, modelClass=Model, options={}):
        """
        options dict: parameters for the dataset
        """
        super().__init__(options=options)

        self.modelClass = modelClass
        self.models = []

        # self.iterationIds = []
        # self.iteration = -1

        # Contains the following keys
        # - min_y_best_locality: minimum of outcomes to keep in best locality searches
        # - min_a_best_locality: minimum of actions to keep in best locality searches
        self.options = options
        self.setOptions()

        # self.spacesHistory = DataEventHistory()
        # self.modelsHistory = DataEventHistory()

    # def gid(self):
    #     if self.moduleParent:
    #         return Serializer.make_gid(self, self.moduleParent.gid())
    #     return Serializer.make_gid(self)

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['models', 'iterationIds', 'iteration']))
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, modelClass=Model, options=None, obj=None):
    #     if 'model' in options:
    #         from ..utils.loaders import DataManager
    #         modelClass = DataManager.loadType(
    #             options['model']['path'], options['model']['type'])
    #     obj = obj if obj else cls(
    #         modelClass=modelClass, options=dict_.get('options', {}))
    #     obj = Module._deserialize(dict_, options=options, obj=obj)
    #     obj = SpaceManager._deserialize(dict_, options=options, obj=obj)
    #     # Operations
    #     models = [obj.modelClass._deserialize(
    #         model, obj, spaces, loadResults=loadResults) for model in dict_.get('models', [])]
    #     return obj

    '''@classmethod
    def clone_without_content(cls, dataset):
        """Function that will copy an other dataset except without any data."""
        result = Dataset([], [], copy.deepcopy(dataset.options))
        for a_s in dataset.actionSpaces:
            result.actionSpaces.append([])
            for a in a_s:
                result.actionSpaces[-1].append(ActionSpace.clone_without_content(a))
        for y in dataset.outcomeSpaces:
            result.outcomeSpaces.append(OutcomeSpace.clone_without_content(y))
        return result'''

    # Models
    def model(self, index):
        return next(s for s in self.models if s.id == index)

    def registerModel(self, model):
        if model not in self.models:
            self.models.append(model)
            self.computeSpaces()

    def unregisterModel(self, model):
        if model in self.models:
            self.models.remove(model)
            self.computeSpaces()

    def replaceModel(self, currentModel, newModel):
        # del self.models[self.models.index(newModel)]
        # self.models[self.models.index(currentModel)] = newModel
        self.unregisterModel(currentModel)
        self.unregisterModel(newModel)
        newModel.continueFrom(currentModel)

    def findModelByOutcomeSpace(self, outcomeSpace, models=None):
        return (self.findModelsByOutcomeSpace(outcomeSpace, models) + [None])[0]

    def findModelsByOutcomeSpace(self, outcomeSpace, models=None):
        models = models if models else self.models
        return [m for m in models if m.coversOutcomeSpaces(outcomeSpace)]

    def findModelsByActionSpace(self, actionSpace, models=None):
        models = models if models else self.models
        return [m for m in models if m.coversActionSpaces(actionSpace)]

    # Graph
    def dependencyGraph(self, models=None):
        models = models if models else self.models

        graph = {}
        for model in models:
            edges = set()
            for space in model.outcomeSpace.flatSpaces:
                edges.update(set(self.findModelsByActionSpace(space, models)))
            graph[model] = tuple(edges)

        return graph

    def isGraphCyclic(self, graph):
        path = set()
        visited = set()

        def visit(node):
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for neighbour in graph.get(node, ()):
                if neighbour in path or visit(neighbour):
                    return True
            path.remove(node)
            return False

        return any(visit(node) for node in graph)

    # Spaces
    def controllableSpaces(self, spaces=None, merge=False):
        return self.__controllableSpaces(True, spaces, merge)

    def nonControllableSpaces(self, spaces=None, merge=False):
        return self.__controllableSpaces(False, spaces, merge)

    def __controllableSpaces(self, controllable, spaces=None, merge=False):
        spaces = spaces if spaces else self.actionExplorationSpaces
        if type(spaces) in (list, set):
            spaces = [space.convertTo(kind=SpaceKind.BASIC)
                      for space in spaces]
        else:
            spaces = spaces.convertTo(kind=SpaceKind.BASIC)
        spaces = [space for space in spaces if self.controllableSpace(
            space) == controllable]
        if merge:
            return self.multiColSpace(spaces)
        return spaces

    def controllableSpace(self, space):
        if space is None or space.null():
            return False
        if space.primitive():
            return True
        m = self.findModelByOutcomeSpace(space)
        return m is not None

    def controlContext(self, goalContext, currentContext):
        if not currentContext:
            return goalContext

        space = goalContext.space
        currentContext = currentContext.projection(space)

        nonControllable = currentContext.projection(
            self.nonControllableSpaces(space, merge=True))
        controllable = goalContext.projection(
            self.controllableSpaces(space, merge=True))
        context = nonControllable.extends(controllable)

        return context

    # Data
    def setData(self, spaces, models, addHistory=True):
        if len(self.spaces) + len(self.models) > 0:
            return
        self.models = models
        self.spaces = spaces
        self.computeSpaces()

        # Add default spaces and models to history
        if addHistory:
            self.spacesHistory.append(self.getIteration(), DataEventKind.ADD,
                                      [("{}".format(s.id), s.serialize()) for s in self.spaces])
            self.modelsHistory.append(self.getIteration(), DataEventKind.ADD,
                                      [("{}".format(m.id), m.serialize()) for m in self.models])

    def setOptions(self):
        if 'min_y_best_locality' not in self.options.keys():
            self.options['min_y_best_locality'] = 5
        if 'min_a_best_locality' not in self.options.keys():
            self.options['min_a_best_locality'] = 4

    '''def create_a_space(self, a_type):
        """When a new length of actions is encountered, creates all action spaces recquired."""
        while a_type[1] >= len(self.actionSpaces[a_type[0]]):
            self.actionSpaces[a_type[0]].append(SingleSpace(self.actionSpaces[a_type[0]][0].bounds, \
                len(self.actionSpaces[a_type[0]])+1, self.actionSpaces[a_type[0]][0].name))'''

    '''def get_space_eventId(self, space, eventId):
        #"""Return outcome id reached at execution id given in outcome space specified."""
        for couple in self.idY[idE]:
            if couple[0] == y_type:
                return couple[1]
        return -1'''

    def goalCompetence(self, y):
        """Compute competence to reach a specific outcome as the distance to nearest neighbour."""
        model = self.findModelByOutcomeSpace(y.space)
        if not model:
            return 0.
        return model.goalCompetence(y)
        '''space = y.get_space()
        _, dist = space.nearestDistance(y, n=1)
        if len(dist) == 0:
            return space.options['out_dist']/space.options['max_dist']
        return dist[0]'''

    '''def get_competence_std_paths(self, paths):
        competence = 1
        std = 0
        dist = 0
        for path in paths:
            for node in path:
                c, s, d = node.model.get_competence_std(node.goal)
                competence *= c
                std += s
                dist += d
        return competence, std, dist

    def normalize(self, vector):
        """Make sure the action is within bounds."""
        for part in vector.get():
            if part.isVector():
                self.normalize(part)
            else:
                space = part.get_space()
                part.value = [max(min(v, space.bounds['max'][i]), space.bounds['min'][i]) for i, v in enumerate(part.value)]'''

    '''def nearest(self, point, n=1):
        """Compute nearest neighbours of an outcome and return ids and performances."""
        return point.space().nearest(point, n=n)'''

    # def nn_a(self, a, a_type, n=1):
    # """Compute nearest neighbours of an action and return ids and distances."""
    # return self.actionSpaces[a_type[0]][a_type[1]].nearest(a, n=n)

    '''def nn_p(self, p, p_type, n=1):
        """Compute nearest neighbours of a procedure and return ids and distances."""
        return self.p_spaces[p_type[0]][p_type[1]].nearest(p, n=n)

    def nn_y_procedural(self, y, task, n=1):
        """Compute nearest neghbours of an outcome among outcomes reached by a procedure only."""
        # get outcome ids of the outcomes reached by a procedure
        ids = np.in1d(self.outcomeSpaces[task].ids, self.idEP)
        all_ids = np.arange(len(self.outcomeSpaces[task].ids))
        ids = all_ids[ids]

        if len(ids) == 0:
            return [], []

        data = np.array(self.outcomeSpaces[task].data)[ids]
        dist = np.sum((data - y)**2, axis=1)/self.outcomeSpaces[task].options['max_dist']
        perf = dist * np.array(self.outcomeSpaces[task].costs)[ids]
        i = range(len(perf))
        i.sort(key=lambda j: perf[j])
        i = i[0:min(len(perf), n)]
        return ids[i], perf[i]

    def refine_procedure(self, p, p_type):
        """Refine a procedure to return the ids of the nearest outcomes reached for both subparts of the procedure."""
        return self.p_spaces[p_type[0]][p_type[1]].nn_procedure(p)'''

    # def plot_actions_hist(self, ax, options):
    #     """Plot number of actions tried by action space."""
    #     width = 0.5
    #     i = 0
    #     x = []
    #     ticks = []
    #     y = []
    #     for a in self.actionSpaces:
    #         for a_s in a:
    #             x.append(i)
    #             y.append(len(a_s.data))
    #             ticks.append(a_s.name + str(a_s.n))
    #             i += 1
    #     ax.bar(x, y, width, color=options['color'])
    #     ax.set_xticks(np.array(x) + width/2.0)
    #     ax.set_xticklabels(ticks)

    # def plot_procedures_hist(self, ax, options):
    #     """Plot number of procedures tried by action space."""
    #     width = 0.5
    #     i = 0
    #     x = []
    #     ticks = []
    #     y = []
    #     for p in self.p_spaces:
    #         for p_s in p:
    #             x.append(i)
    #             y.append(len(p_s.data))
    #             ticks.append(p_s.name)
    #             i += 1
    #     ax.bar(x, y, width, color=options['color'])
    #     ax.set_xticks(np.array(x) + width/2.0)
    #     ax.set_xticklabels(ticks)

    # def plot_actions_hist_task(self, task, ax, options):
    #     """Plot number of actions tried by action space that reached given outcome space."""
    #     nb_a = []
    #     width = 0.5
    #     i = 0
    #     x = []
    #     ticks = []
    #     for a in self.actionSpaces:
    #         nb_a.append([])
    #         for a_s in a:
    #             nb_a[-1].append(0)
    #             x.append(i)
    #             ticks.append(a_s.name + str(a_s.n))
    #             i += 1
    #     idE = self.outcomeSpaces[task].ids
    #     for i in idE:
    #         a_type = self.idA[i][0]
    #         nb_a[a_type[0]][a_type[1]] += 1
    #     y = []
    #     for nearest in nb_a:
    #         for n in nearest:
    #             y.append(n)
    #     ax.bar(x, y, width, color=options['color'])
    #     ax.set_xticks(np.array(x) + width/2.0)
    #     ax.set_xticklabels(ticks)

    # def plot_procedures_hist_task(self, task, ax, options):
    #     """Plot number of procedures tried by action space that reached given outcome space."""
    #     nb_p = []
    #     width = 0.5
    #     i = 0
    #     x = []
    #     ticks = []
    #     for p in self.p_spaces:
    #         nb_p.append([])
    #         for p_s in p:
    #             nb_p[-1].append(0)
    #             x.append(i)
    #             ticks.append(p_s.name)
    #             i += 1
    #     idE = self.outcomeSpaces[task].ids
    #     for i in idE:
    #         if not self.idP[i]:
    #             continue
    #         p_type = self.idP[i][0]
    #         nb_p[p_type[0]][p_type[1]] += 1
    #     y = []
    #     for nearest in nb_p:
    #         for n in nearest:
    #             y.append(n)
    #     ax.bar(x, y, width, color=options['color'])
    #     ax.set_xticks(np.array(x) + width/2.0)
    #     ax.set_xticklabels(ticks)

    # def plot_outcomes(self, task, ax, options):
    #     """Plot outcomes in given outcome space;"""
    #     return self.outcomeSpaces[task].plot_outcomes(ax, options)

    # def plot_outcomes_atype(self, task, a_type, ax, options):
    #     """Plot outcomes reached by the given action space on the given outcome space."""
    #     if self.outcomeSpaces[task].dim == 1:
    #         pp = np.array(self.outcomeSpaces[task].data)
    #         idY = np.array(self.outcomeSpaces[task].ids)
    #         idA = np.array(self.actionSpaces[a_type[0]][a_type[1]].ids)
    #         coms = np.in1d(idY, idA)
    #         if len(pp) > 0:
    #             pp = pp[coms, :]
    #             return ax.plot(pp, np.zeros_like(pp), marker=options['marker'], color=options['color'], linestyle=options['linestyle'])
    #     elif self.outcomeSpaces[task].dim == 2:
    #         pp = np.array(self.outcomeSpaces[task].data)
    #         idY = np.array(self.outcomeSpaces[task].ids)
    #         idA = np.array(self.actionSpaces[a_type[0]][a_type[1]].ids)
    #         coms = np.in1d(idY, idA)
    #         if len(pp) > 0:
    #             pp = pp[coms, :]
    #             return ax.plot(pp[:, 0], pp[:, 1], marker=options['marker'], color=options['color'], linestyle=options['linestyle'])
    #     elif self.outcomeSpaces[task].dim == 3:
    #         pp = np.array(self.outcomeSpaces[task].data)
    #         idY = np.array(self.outcomeSpaces[task].ids)
    #         idA = np.array(self.actionSpaces[a_type[0]][a_type[1]].ids)
    #         coms = np.in1d(idY, idA)
    #         if len(pp) > 0:
    #             pp = pp[coms, :]
    #             # marker= '.', color='b'
    #             return ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], marker=options['marker'], color=options['color'])
    #     else:
    #         pass

    # def plot_actions(self, a_type, ax, options=None):
    #     pp = np.array(self.actionSpaces[a_type][0].data)
    #     if len(pp) > 0:
    #         return ax.plot(pp[:, 0], pp[:, 1], marker=options['marker'], color=options['color'], linestyle=options['linestyle'])

    # #### The following functions are here to ease the use of Visualizers

    # '''def get_plot_actions_visualizer(self, a_type, prefix=""):
    #     dico = {}
    #     dico['limits'] = {'min': self.actionSpaces[a_type][0].bounds['min'], \
    #         'max': self.actionSpaces[a_type][0].bounds['max']}
    #     dico['title'] = prefix + "Points reached actions " + str(a_type)
    #     dico['color'] = ['k']
    #     dico['marker'] = ['.']
    #     dico['linestyle'] = ['None']
    #     dico['legend'] = ['normal']
    #     dico['plots'] = [lambda fig, ax, options: self.plot_actions(a_type, ax, options)]

    #     return dico'''

    # def get_model_reached_visualizer(self, model, prefix=""):
    #     task = self.outcomeSpaces.index(model.outcomes_space)
    #     a_type = self.actionSpaces.index([model.actionSpace])
    #     dict_outcomes = self.get_reached_one_atype_visualizer(
    #         task, (a_type, 0), prefix)

    #     def onclick(event):
    #         ydata = [d.tolist() for d in model.outcomes_space.data]
    #         ids = []
    #         for point in event['points']:
    #             for i, p in enumerate(ydata):
    #                 if point[0] == p[0] and point[1] == p[1]:
    #                     ids.append(model.outcomes_space.ids[i])
    #         adata = [d[0].tolist() for d in zip(model.actionSpace.data,
    #                                             model.actionSpace.ids) if d[1] in ids]

    #         response = {}
    #         response['highlight'] = (1, adata)
    #         return response
    #         #print(point)
    #         #print(point.all() in ydata)
    #         #print(event['points'])
    #         #print(model.outcomes_space.data)
    #     dict_outcomes['onclick'] = onclick

    #     return [dict_outcomes, self.get_plot_actions_visualizer(a_type, prefix)]

    # def get_reached_one_atype_visualizer(self, task, a_type, prefix=""):
    #     """Return a dictionary used to visualize outcomes reached by one action space."""
    #     dico = {}
    #     dico['limits'] = {'min': self.outcomeSpaces[task].bounds['min'],
    #                       'max': self.outcomeSpaces[task].bounds['max']}
    #     dico['title'] = prefix + "Points reached task " + str(task) + " with " + self.actionSpaces[a_type[0]][a_type[1]].name + \
    #         str(self.actionSpaces[a_type[0]][a_type[1]].n)
    #     dico['color'] = ['k']
    #     dico['marker'] = ['.']
    #     dico['linestyle'] = ['None']
    #     dico['legend'] = [self.actionSpaces[a_type[0]][a_type[1]].name]
    #     dico['plots'] = [lambda fig, ax,
    #                      options: self.plot_outcomes_atype(task, a_type, ax, options)]

    #     return dico

    # def get_reached_atype_visualizer(self, task, prefix=""):
    #     """Return a dictionary used to visualize outcomes reached for each action space used."""
    #     cmap = plt.cm.jet
    #     colors = []
    #     markers = []
    #     lines = []
    #     legends = []
    #     plots = []
    #     n_a_spaces = 0
    #     n = 0
    #     for i in range(len(self.actionSpaces)):
    #         n_a_spaces += len(self.actionSpaces[i])
    #     for i in range(len(self.actionSpaces)):
    #         for j in range(len(self.actionSpaces[i])):
    #             if n_a_spaces == 1:
    #                 colors.append('k')
    #             else:
    #                 colors.append(
    #                     cmap(int(math.floor(float(n)*float(cmap.N - 1)/float(n_a_spaces-1)))))
    #             markers.append('.')
    #             lines.append('None')
    #             legends.append(
    #                 self.actionSpaces[i][j].name + str(self.actionSpaces[i][j].n))
    #             plots.append(lambda fig, ax, options, i=i, j=j: self.plot_outcomes_atype(
    #                 task, (i, j), ax, options))
    #             n += 1
    #     dico = {}
    #     dico['limits'] = {'min': self.outcomeSpaces[task].bounds['min'],
    #                       'max': self.outcomeSpaces[task].bounds['max']}
    #     dico['title'] = prefix + "Points reached task " + str(task)
    #     dico['color'] = colors
    #     dico['marker'] = markers
    #     dico['linestyle'] = lines
    #     dico['legend'] = legends
    #     dico['plots'] = plots

    #     return dico

    # def get_reached_visualizer(self, task, prefix=""):
    #     """Return a dictionary used to visualize outcomes reached for the specified outcome space."""
    #     dico = getVisual(
    #         [lambda fig, ax, options: self.plot_outcomes(task, ax, options)],
    #         minimum=self.outcomeSpaces[task].bounds['min'],
    #         maximum=self.outcomeSpaces[task].bounds['max'],
    #         title=prefix + "Points reached task " + str(task))
    #     dico['title'] = prefix + "Points reached task " + str(task)
    #     dico['color'] = ['k']
    #     dico['marker'] = ['.']
    #     dico['linestyle'] = ['None']
    #     dico['legend'] = []
    #     dico['plots'] = [lambda fig, ax,
    #                      options: self.plot_outcomes(task, ax, options)]

    #     return dico

    # def get_actionspaces_visualizer(self, prefix=""):
    #     """Return a dictionary used to visualize the number of actions tried per action space."""
    #     dico = {}
    #     dico['limits'] = {'min': [None, None], 'max': [None, None]}
    #     dico['title'] = "Action spaces distribution " + prefix
    #     dico['color'] = ['b']
    #     dico['marker'] = [None]
    #     dico['linestyle'] = ['None']
    #     dico['legend'] = []
    #     dico['plots'] = [lambda fig, ax,
    #                      options: self.plot_actions_hist(ax, options)]

    #     return dico

    # def get_actionspaces_task_visualizer(self, task, prefix=""):
    #     """Return a dictionary used to visualize the number of actions tried per action space for given outcome space."""
    #     dico = {}
    #     dico['limits'] = {'min': [None, None], 'max': [None, None]}
    #     dico['title'] = "Action spaces distribution for task " + \
    #         str(task) + " for " + prefix
    #     dico['color'] = ['b']
    #     dico['marker'] = [None]
    #     dico['linestyle'] = ['None']
    #     dico['legend'] = []
    #     dico['plots'] = [lambda fig, ax,
    #                      options: self.plot_actions_hist_task(ax, task, options)]

    #     return dico

    # def get_proceduralspaces_visualizer(self, prefix=""):
    #     """Return a dictionary used to visualize the number of procedures tried per procedure space."""
    #     dico = {}
    #     dico['limits'] = {'min': [None, None], 'max': [None, None]}
    #     dico['title'] = "Procedural spaces distribution " + prefix
    #     dico['color'] = ['b']
    #     dico['marker'] = [None]
    #     dico['linestyle'] = ['None']
    #     dico['legend'] = []
    #     dico['plots'] = [lambda fig, ax,
    #                      options: self.plot_procedures_hist(ax, options)]

    #     return dico

    # def get_proceduralspaces_task_visualizer(self, task, prefix=""):
    #     """Return a dictionary used to visualize the number of procedures tried per procedure space for given outcome space."""
    #     dico = {}
    #     dico['limits'] = {'min': [None, None], 'max': [None, None]}
    #     dico['title'] = "Procedural spaces distribution for task " + \
    #         str(task) + " for " + prefix
    #     dico['color'] = ['b']
    #     dico['marker'] = [None]
    #     dico['linestyle'] = ['None']
    #     dico['legend'] = []
    #     dico['plots'] = [lambda fig, ax,
    #                      options: self.plot_procedures_hist_task(ax, task, options)]

    #     return dico

    # # Visual
    # def plotDependencyGraph(self, models=None):
    #     data = self.dependencyGraph(models)

    #     g = graphviz.Digraph('Models')
    #     for node, _ in data.items():
    #         g.node('Model {}'.format(node.id))
    #     for node, edges in data.items():
    #         for edge in edges:
    #             g.edge('Model {}'.format(node.id), 'Model {}'.format(edge.id))

    #     return g

    # # Api
    # def apiget_hierarchy(self, range_=(-1, -1)):
    #     return {'spaces': self.spacesHistory.get_range(range_), 'models': self.modelsHistory.get_range(range_)}

    # def apiget_space_points(self, space_id, range_=(-1, -1)):
    #     space = self.get_space(space_id)
    #     eventIds = iterrange(self.iterationIds, range_)
    #     iterations = [item for sublist in eventIds for item in [
    #         sublist[0]] * (len(sublist) - 1)]
    #     eventIds = [item for sublist in eventIds for item in sublist[1:]]
    #     data = list(zip(iterations, space.apiGetPoints(eventIds)))
    #     return {'data': data}

    # def apiget_model(self, model_id, range_=(-1, -1)):
    #     return self.model(model_id).apiget(range_)
