import os
import random
from level_generator import ParamGenerator
import glob
from baselines.a2c.utils import make_path


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
                 'pcg-progressive']

    @staticmethod
    def get_selector(selector_name, game, path):
        if selector_name is not None:
            make_path(path)
            path = os.path.realpath(path)
            if selector_name == "random-all":
                return RandomSelector(path, game, [0, 1, 2, 3, 4])
            elif selector_name == "random-0123":
                return RandomSelector(path, game, [0, 1, 2, 3])
            elif selector_name.startswith('random-'):
                difficulty = float(selector_name.split('random-')[1]) * 0.1
                return RandomWithDifSelector(path, game, difficulty)
            elif selector_name == "pcg-random":
                return RandomPCGSelector(path, game)
            elif selector_name.startswith('pcg-random-'):
                difficulty = float(selector_name.split('pcg-random-')[1]) * 0.1
                return RandomWithDifPCGSelector(path, game, difficulty)
            elif selector_name == "pcg-progressive":
                return ProgressivePCGSelector(path, game)
        return None

    def __init__(self, dir, game):
        self.dir = dir
        if self.dir[-1] != "/":
            self.dir += "/"
        self.game = game.lower()

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

game_sizes = {
    "aliens": [30, 11],
    "zelda": [13, 9],
    "boulderdash": [26, 13]
}


class RandomSelector(LevelSelector):

    def __init__(self, dir, game, lvl_ids):
        super().__init__(dir, game)
        self.lvl_ids = lvl_ids

    def get_level(self):
        lvl_id = random.choice(self.lvl_ids)
        return lvl_id

    def report(self, level_id, win):
        pass


class RandomWithDifSelector(LevelSelector):

    def __init__(self, dir, game, difficulty):
        super().__init__(dir, game)
        self.difficulty = difficulty
        path = os.path.dirname(os.path.realpath(__file__)) + "/data/test-levels/zelda/" + str(int(difficulty * 10)) + "/"
        self.levels = [filename for filename in glob.iglob(path + '*')]

    def get_level(self):
        return random.choice(self.levels)

    def report(self, level_id, win):
        pass


class RandomPCGSelector(LevelSelector):

    def __init__(self, dir, game):
        super().__init__(dir, game)
        size = game_sizes[game]
        self.generator = ParamGenerator(self.dir, self.game, width=size[0], height=size[1])

    def get_level(self):
        return self.generator.generate()

    def report(self, level_id, win):
        pass


class RandomWithDifPCGSelector(LevelSelector):

    def __init__(self, dir, game, difficulty):
        super().__init__(dir, game)
        self.difficulty = difficulty
        size = game_sizes[game]
        self.generator = ParamGenerator(self.dir, self.game, width=size[0], height=size[1])

    def get_level(self):
        return self.generator.generate([self.difficulty], difficulty=True)

    def report(self, level_id, win):
        pass


class ProgressivePCGSelector(LevelSelector):
    '''
    TODO: Shared object across workers.
    '''
    def __init__(self, dir, game, alpha=0.01):
        super().__init__(dir, game)
        size = game_sizes[game]
        self.generator = ParamGenerator(self.dir, self.game, width=size[0], height=size[1])
        self.difficulty = 0
        self.alpha = alpha

    def get_level(self):
        return self.generator.generate([self.difficulty], difficulty=True)

    def report(self, level_id, win):
        if win:
            self.difficulty = min(1.0, self.difficulty + self.alpha)
        else:
            self.difficulty = max(0.0, self.difficulty - self.alpha)

#sel = RandomPCGSelector("./", "zelda")
#level = sel.get_level()
#print("Playing on level " + level)
#sel.report(level, False)
