
#!/bin/bash

aggregated_result_file="logs/adaicl_adaptive.txt"

model_name="gpt-neo-1.3B"
strategy=(ada_icl_default random ada_icl votek)
dataset=(sst2 rte mrpc mnli amazon trec ag_news)

for s in 0 
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in $(seq 6 6)
    do

        for m in $(seq 0 3)
        do
            printf "%10s\t" $m >> $aggregated_result_file
            CUDA_VISIBLE_DEVICES=$((m+4)) python main_adaptive_phases.py  --phases 1  --few_shot 5 --task_name ${dataset[t]} --selective_annotation_method ${strategy[m]} --model_cache_dir "models" --data_cache_dir "datasets" --output_dir "outputs" --annotation_size 100 --model_name $model_name --seed $s --init "cluster"  --sample_k >> $aggregated_result_file &
        done

        printf "\n" >> $aggregated_result_file    
    done
done
