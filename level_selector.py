import random
from level_generator import ParamGenerator


class LevelSelector(object):

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


class RandomPCGSelector(LevelSelector):

    def __init__(self, dir, game):
        super().__init__(dir, game)
        size = game_sizes[game]
        self.generator = ParamGenerator(self.dir, self.game, width=size[0], height=size[1])

    def get_level(self):
        return self.generator.generate()

    def report(self, level_id, win):
        pass


class ProgressivePCGSelector(LevelSelector):

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
