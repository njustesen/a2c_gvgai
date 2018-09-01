#!/bin/sh
NUM_ENVS=8
RUNS=30
EXPERIMENT_NAME="zelda-ls-pcg-random-10"
EXPERIMENT_ID1="5febfd8c-a0d6-11e8-ae14-000d3a1dca10"
EXPERIMENT_ID2="6fd5f074-9bf2-11e8-ade4-000d3a1dca10"
EXPERIMENT_ID3="58de77f4-a0d6-11e8-ae14-000d3a1dca10"
EXPERIMENT_ID4="59fab7c6-9bf2-11e8-ade3-000d3a1dca10"

for (( i=0; i <= 4; ++i ))
do
    python eval.py --game zelda --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
    python eval.py --game zelda --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
    python eval.py --game zelda --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
    python eval.py --game zelda --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
done


python eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
python eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
python eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
python eval.py --game zelda --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4

python eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
python eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
python eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
python eval.py --game zelda --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
