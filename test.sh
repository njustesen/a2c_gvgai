#!/bin/sh
NUM_ENVS=8
RUNS=30
EXPERIMENT_NAME="boulderdash-ls-pcg-progressive-fixed"
EXPERIMENT_ID1="051ed6a0-58d2-11e8-a381-000d3a162dc9"
EXPERIMENT_ID2="64dacd2c-7594-11e8-8944-000d3a606f4e"
EXPERIMENT_ID3="1263205a-58d2-11e8-a381-000d3a162dc9"
EXPERIMENT_ID4="43782882-7594-11e8-bc14-000d3a1d5440"
GAME="boulderdash"

for (( i=0; i <= 4; ++i ))
do
    python eval.py --game $GAME --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
    python eval.py --game $GAME --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
    python eval.py --game $GAME --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
    python eval.py --game $GAME --selector seq-human-$i --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
done


python eval.py --game $GAME --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
python eval.py --game $GAME --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
python eval.py --game $GAME --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
python eval.py --game $GAME --selector seq-5 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4

python eval.py --game $GAME --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID1
python eval.py --game $GAME --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID2
python eval.py --game $GAME --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID3
python eval.py --game $GAME --selector seq-10 --num-envs $NUM_ENVS --runs $RUNS --experiment-name $EXPERIMENT_NAME --experiment-id $EXPERIMENT_ID4
