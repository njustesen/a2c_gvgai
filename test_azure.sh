#!/bin/sh
NUM_ENVS=2
RUNS=4
EXPERIMENT_NAME="zelda-lvl-0"
EXPERIMENT_ID1="88c3e2ba-5883-11e8-bbf6-6c4008b68262"
EXPERIMENT_ID2="88c3e2ba-5883-11e8-bbf6-6c4008b68262"
EXPERIMENT_ID3="88c3e2ba-5883-11e8-bbf6-6c4008b68262"
EXPERIMENT_ID4="88c3e2ba-5883-11e8-bbf6-6c4008b68262"

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