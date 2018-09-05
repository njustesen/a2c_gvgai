#!/bin/bash
NUM_ENVS=14
RUNS=30
EXPERIMENT_NAME="zelda-ls-pcg-progressive-fixed"
EXPERIMENT_ID1="3f802970-9d88-11e8-877d-000d3a606f4e"
EXPERIMENT_ID2="cfd1eb06-9bf0-11e8-8557-000d3a606f4e"
EXPERIMENT_ID3="f6bdd36a-9bf0-11e8-855a-000d3a606f4e"
EXPERIMENT_ID4="f68e7a6c-9d61-11e8-8722-000d3a606f4e"

for (( i=0; i <= 4; ++i ))
do
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
    sudo python3 eval.py --game zelda --level $i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
done

sudo python3 eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
sudo python3 eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
sudo python3 eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
sudo python3 eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4

sudo python3 eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
sudo python3 eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
sudo python3 eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
sudo python3 eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
