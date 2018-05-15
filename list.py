import tensorflow as tf
import argparse

from level_selector import *
from model import Model
from runner import Runner
from env import *

from baselines.a2c.utils import make_path
from baselines.a2c.policies import CnnPolicy

from baselines.common import set_global_seeds
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def main():

    exp_path = './results/*/'
    for exp_folder in glob.iglob(exp_path):

        # Experiment name
        exp_name = exp_folder.split('/')[-2]
        print(exp_name)

        path = os.path.join(exp_folder, 'models/*/')
        for folder in glob.iglob(path):
            steps = 0
            exp_id = folder.split('/')[-2]

            for model_meta_name in glob.iglob(folder + '/*.meta'):
                s = int(model_meta_name.split('.meta')[0].split('/')[-1].split("-")[1])
                if s >= steps:
                    steps = s

            print('\t' + exp_id + '\t' + str(steps))


if __name__ == '__main__':
    main()
