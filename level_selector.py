import os
import random
from level_generator import ParamGenerator
import glob
from baselines.a2c.utils import make_path
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager


class LevelSelector(object):

    available = ['random-all',
                 'random-0123',
                 'random-0',
                 'random-1',
                 'random-2',
                 'random-3',
                 'random-4',
                 'random-5',
                 'random-6',
                 'random-7',
                 'random-8',
                 'random-9',
                 'random-10',
                 'seq-0',
                 'seq-1',
                 'seq-2',
                 'seq-3',
                 'seq-4',
                 'seq-5',
                 'seq-6',
                 'seq-7',
                 'seq-8',
                 'seq-9',
                 'seq-10',
                 'pcg-random',
                 'pcg-random-0',
                 'pcg-random-1',
                 'pcg-random-2',
                 'pcg-random-3',
                 'pcg-random-4',
                 'pcg-random-5',
                 'pcg-random-6',
                 'pcg-random-7',
                 'pcg-random-8',
                 'pcg-random-9',
                 'pcg-random-10',
                 'pcg-progressive',
                 'pcg-progressive-fixed']

    @staticmethod
    def get_selector(selector_name, game, path, fixed=False):

        # Register classes for sharing across procs
        for c in [RandomSelector, RandomWithDifSelector, RandomPCGSelector, RandomWithDifPCGSelector, ProgressivePCGSelector, SequentialSelector]:
            BaseManager.register(c.__name__, c)
        manager = BaseManager()
        manager.start()

        # Determine selector
        if selector_name is not None:
            make_path(path)
            path = os.path.realpath(path)
            if selector_name == "random-all":
                selector = manager.RandomSelector(path, game, [0, 1, 2, 3, 4])
            elif selector_name == "random-0123":
                selector = manager.RandomSelector(path, game, [0, 1, 2, 3])
            elif selector_name.startswith('random-'):
                difficulty = float(selector_name.split('random-')[1]) * 0.1
                selector = manager.RandomWithDifSelector(path, game, difficulty)
            elif selector_name.startswith('seq-'):
                difficulty = float(selector_name.split('seq-')[1]) * 0.1
                selector = manager.SequentialSelector(path, game, difficulty)
            elif selector_name == "pcg-random":
                selector = manager.RandomPCGSelector(path, game)
            elif selector_name.startswith('pcg-random-'):
                difficulty = float(selector_name.split('pcg-random-')[1]) * 0.1
                selector = manager.RandomWithDifPCGSelector(path, game, difficulty, fixed=fixed)
            elif selector_name == "pcg-progressive":
                selector = manager.ProgressivePCGSelector(path, game)
            elif selector_name == "pcg-progressive-fixed":
                selector = manager.ProgressivePCGSelector(path, game, upper_limit=False)
            else:
                raise Exception("Unknown level selector: + " + selector_name)
        else:
            return None

        return selector

    def __init__(self, dir, game):
        self.dir = dir
        if self.dir[-1] != "/":
            self.dir += "/"
        self.game = game.lower()

    def get_game(self):
        return self.game

    def get_level(self):
        '''
        :return: the id and path of the next level to be evaluated.
        '''
        raise NotImplementedError

    def report(self, level_id, win):
        '''
        :param level_id: id of the level in which the score was achieved
        :param score: the in-game score of the level
        '''
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

game_sizes = {
    "aliens": [30, 11],
    "zelda": [13, 9],
    "boulderdash": [26, 13],
    "frogs": [28, 11],
    "solarfox": [10,11]
}


class SequentialSelector(LevelSelector):

    def __init__(self, dir, game, difficulty):
        super().__init__(dir, game)
        self.difficulty = difficulty
        path = os.path.dirname(os.path.realpath(__file__)) + "/data/test-levels/" + game + "/" + str(int(difficulty * 10)) + "/"
        self.levels = [filename for filename in glob.iglob(path + '*')]
        self.idx = 0

    def get_level(self):
        level = self.levels[self.idx]
        self.idx = self.idx + 1
        if self.idx >= len(self.levels):
            self.idx = 0
        return level

    def report(self, level_id, win):
        pass

    def get_info(self):
        return ""


class RandomSelector(LevelSelector):

    def __init__(self, dir, game, lvl_ids):
        super().__init__(dir, game)
        self.lvl_ids = lvl_ids

    def get_level(self):
        lvl_id = random.choice(self.lvl_ids)
        return lvl_id

    def report(self, level_id, win):
        pass

    def get_info(self):
        return ""


class RandomWithDifSelector(LevelSelector):

    def __init__(self, dir, game, difficulty):
        super().__init__(dir, game)
        self.difficulty = difficulty
        path = os.path.dirname(os.path.realpath(__file__)) + "/data/test-levels/" + game + "/" + str(int(difficulty * 10)) + "/"
        self.levels = [filename for filename in glob.iglob(path + '*')]

    def get_level(self):
        return random.choice(self.levels)

    def report(self, level_id, win):
        pass

    def get_info(self):
        return ""


class RandomPCGSelector(LevelSelector):

    def __init__(self, dir, game):
        super().__init__(dir, game)
        size = game_sizes[game]
        self.generator = ParamGenerator(self.dir, self.game, width=size[0], height=size[1])

    def get_level(self):
        return self.generator.generate()

    def report(self, level_id, win):
        pass

    def get_info(self):
        return ""


class RandomWithDifPCGSelector(LevelSelector):

    def __init__(self, dir, game, difficulty, fixed=False):
        super().__init__(dir, game)
        self.difficulty = difficulty
        self.fixed = fixed
        size = game_sizes[game]
        self.generator = ParamGenerator(self.dir, self.game, width=size[0], height=size[1])
        self.last_level = None

    def get_level(self):
        if self.fixed and self.last_level is not None:
            return self.last_level
        self.last_level = self.generator.generate([self.difficulty], difficulty=True)
        return self.last_level

    def report(self, level_id, win):
        pass

    def get_info(self):
        return ""


class ProgressivePCGSelector(LevelSelector):
    '''
    TODO: Shared object across workers.
    '''
    def __init__(self, dir, game, alpha=0.01, upper_limit=True):
        super().__init__(dir, game)
        size = game_sizes[game]
        self.generator = ParamGenerator(self.dir, self.game, width=size[0], height=size[1])
        self.difficulty = 0
        self.alpha = alpha
        self.upper_limit = upper_limit

    def get_level(self):
        if not self.upper_limit:
            return self.generator.generate([self.difficulty], difficulty=True)

        dif = random.uniform(0.0, self.difficulty)
        return self.generator.generate([dif], difficulty=True)

    def report(self, level_id, win):
        if win:
            self.difficulty = min(1.0, self.difficulty + self.alpha)
        else:
            self.difficulty = max(0.0, self.difficulty - self.alpha)

    def get_info(self):
        return str(self.difficulty)

'''
for i in range(11):
    for x in range(10):
        make_path("./data/test-levels/boulderdash/" + str(i) + "/")
        sel = ProgressivePCGSelector("./data/test-levels/boulderdash/" + str(i) + "/", "boulderdash", upper_limit=False)
        sel.difficulty = i*0.1
        level = sel.get_level()
'''
#print("Playing on level " + level)
#sel.report(level, False)
