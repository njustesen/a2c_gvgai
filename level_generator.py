import os
import uuid


class LevelGenerator(object):
    '''
    If this constructor is overridden, call it with super().__init__(dir, game) from the subclass.
    '''

    def __init__(self, dir, game):
        self.dir = dir
        if self.dir[-1] != "/":
            self.dir += "/"
        self.game = game.lower()
        
    def generate(self):
        '''
        :return: the id and path of the next level to be evaluated.
        '''
        raise NotImplementedError


class ParamGenerator(LevelGenerator):

    def __init__(self, dir, game, width, height):
        super().__init__(dir, game)
        self.width = width
        self.height = height
        self.script = os.path.dirname(os.path.realpath(__file__)) + '/lib/gvgai_generator/app_v2.js'

    def generate(self, params=[], difficulty=None):
        name = self.game + "_" + str(uuid.uuid1())
        if difficulty is not None:
            name += "_dif" + str(round(params[0], 2))
            params = ["difficulty"] + params + [self.width, self.height]
        else:
            params = [self.width, self.height] + params
        params = [str(param) for param in params]
        param_str = " ".join(params)
        file = self.dir + name + ".txt"
        os.system("node " + self.script + " " + self.game + " " + file + " " + param_str)
        path = os.path.abspath(file)
        return path

for d in [0,1,2,3,4,5,6,7,8,9,10]:
    gen = ParamGenerator("./data/test-levels/zelda/{}/".format(d), "zelda", 13, 9)
    for i in range(10):
        level = gen.generate([(d/10.0)], difficulty=True)
#print(level)
print("Done")