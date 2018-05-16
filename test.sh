#!/bin/sh
NUM_ENVS=10
RUNS=100

python eval.py --game zelda --level 0 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --level 1 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --level 2 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --level 3 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --level 4 --num-envs $NUM_ENVS --runs $RUNS

python eval.py --game zelda --selector seq-0 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-1 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-2 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-3 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-4 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-6 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-7 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-9 --num-envs $NUM_ENVS --runs $RUNS
python eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS