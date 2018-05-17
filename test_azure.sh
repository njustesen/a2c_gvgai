#!/bin/bash
NUM_ENVS=15
RUNS=30
EXPERIMENT_NAME="zelda-ls-pcg-random-10"
EXPERIMENT_ID1="1995f8da-572d-11e8-80a9-fa163e46f56f"
EXPERIMENT_ID2="8c7177d2-5663-11e8-8079-fa163e46f56f"
EXPERIMENT_ID3="65246228-53b3-11e8-a481-2c4d5445a6f1"
EXPERIMENT_ID4="b1f6410c-57cd-11e8-80e0-fa163e46f56f"

for (( i=0; i <= 4; ++i ))
do
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
done

for (( i=0; i <= 10; ++i ))
do
    sudo python3 eval.py --game zelda --selector seq-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
    sudo python3 eval.py --game zelda --selector seq-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
    sudo python3 eval.py --game zelda --selector seq-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
    sudo python3 eval.py --game zelda --selector seq-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
done
