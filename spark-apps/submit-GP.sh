#!/bin/bash
if [ $# -eq 3 ]; then
    spark-submit \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores 2 \
        --num-executors 5 \
        --executor-memory 512m \
        --conf spark.yarn.executor.memoryOverhead=30m \
        --conf spark.driver.memory=512m \
        --conf spark.driver.cores=3 \
        higgsSparkNRuns.py $1 $2 $3    
    exit
else
    echo -e "\nUsage: submit-GP.sh nb_runs population_size nb_generations\n"
fi