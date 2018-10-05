# A2C for GVG-AI

## Install
1. Install gym
2. Install gvgai-gym
3. Install baselines (due to a recent update it has to be an earlier version from before August - We hope to fix this soon in our repo so it is independant)
4. Run `sh lib/gvgai_generator/install.sh` to install the level generator.
5. Run `pip install brewer2mpl`

## How to Train
```
usage: a2c.py [-h] [--policy {cnn,lstm,lnlstm}]
              [--lrschedule {constant,linear}] [--num-envs NUM_ENVS]
              [--num-timesteps NUM_TIMESTEPS] [--game GAME] [--seed SEED]
              [--save-interval SAVE_INTERVAL] [--level LEVEL] [--fixed]
              [--selector {None,random-all,random-0123,random-0,random-1,random-2,random-3,random-4,random-5,random-6,random-7,random-8,random-9,random-10,pcg-random,pcg-random-0,pcg-random-1,pcg-random-2,pcg-random-3,pcg-random-4,pcg-random-5,pcg-random-6,pcg-random-7,pcg-random-8,pcg-random-9,pcg-random-10,pcg-progressive}]

optional arguments:
  -h, --help            show this help message and exit
  --policy {cnn,lstm,lnlstm}
                        Policy architecture (default: cnn)
  --lrschedule {constant,linear}
                        Learning rate schedule (default: constant)
  --num-envs NUM_ENVS   Number of environments/workers to run in parallel
                        (default=12) (default: 12)
  --num-timesteps NUM_TIMESTEPS  
                        Number of timesteps to train the model
  --game GAME           Game name (default=zelda) (default: zelda)
  --seed SEED           RNG seed (default: 0)
  --save-interval SAVE_INTERVAL
  --resume EXP_ID       The experiment id to resume (default: None)
                        Model saving interval in steps (default: 1000000)
  --level LEVEL         Id of existing GVG-AI level (integer) to train on (default: 0)
  --fixed               A hack to enable a selector to always use the first generated level. Should be removed asap (default: False)
  --selector {
      None,                       Will use a fixed level determined by the level argument
      random-all,                 Samples levels from the existing GVG-AI levels
      random-0123,                Samples levels from the existing GVG-AI levels, except level 4
      random-<difficulty>,        Samples levels from a pre-generated set of 10 levels with a difficulty level of <difficulty> from 0 to 10
      pcg-random,                 Generates levels with random parameters
      pcg-random-<difficulty>,    Generates levels with a difficulty level of <difficulty> from 0 to 10
      pcg-progressive             Generates levels with increasing or decreasing difficulty level if games are won or lost.
    }
                        Level selector to use in training - will ignore the
                        level argument if set (default: None) (default: None)
```

Examples:
```
python a2c.py --game aliens --level 4
```
```
python a2c.py --game boulderdash --selector pcg-random-3
```
```
python a2c.py --game zelda --selector pcg-random-3 --resume 90fe9d4c-56e4-11e8-a58c-6c4008b68262 --num-timesteps 40000000
```

## How to evaluate
eval.py will evaluate all models in an experiment folder.

```
usage: eval.py [-h] [--policy {cnn,lstm,lnlstm}] [--runs RUNS]
               [--num-envs NUM_ENVS] [--game GAME] [--seed SEED]
               [--experiment-name EXPERIMENT_NAME] [--level LEVEL]
               [--level-selector {None,random-all,random-0123,random-0,random-1,random-2,random-3,random-4,random-5,random-6,random-7,random-8,random-9,random-10,pcg-random,pcg-random-0,pcg-random-1,pcg-random-2,pcg-random-3,pcg-random-4,pcg-random-5,pcg-random-6,pcg-random-7,pcg-random-8,pcg-random-9,pcg-random-10,pcg-progressive}]
               [--render]

optional arguments:
  -h, --help            show this help message and exit
  --policy {cnn,lstm,lnlstm}
                        Policy architecture (default: cnn)
  --runs RUNS           Number of runs for each model (default: 100)
  --num-envs NUM_ENVS   Number of environments/workers to run in parallel (default: 12)
  --game GAME           Game name (default=zelda) (default: zelda)
  --seed SEED           RNG seed (default: 0)
  --experiment-name EXPERIMENT_NAME
                        Name of the experiments to evaluate (name of subfolder in .results/) e.g. zelda-ls-pcg-random (default: zelda-lvl-0)
  --level LEVEL         Fixed existing GVG-AI level to test on (default: 0)
  --selector {...}      Same as a2c.py. Will evaluate on levels selected by the selector. Will ignore the level argument.
  --render              Render screen (default: False) (default: False)
  ```
  
Examples:
```
python eval.py --game aliens --experiment-name zelda-lvl-0 --level 2
```
```
python eval.py --game aliens --experiment-name zelda-lvl-0 --selector pcg-random
```
  
## Plot
Simply run the following to produce plots for all experiments in .results/
```
python plot.py
```

## List experiments
```
python list.py
```
  
## Transfer Experiments between Servers
You can either:
1. Move a subfolder within `.results/` from server A to `.resuts/` on server B. This will copy all experiments of that type.
2. Move individual folders within `./results/<experiments-name>/models/` and `./results/<experiments-name>/logs/` from A to B.
