#!/bin/bash

start_time=$(date +%s)

GEN_CHEM_1D=../../../generative_chemistry_1D

which python

python $GEN_CHEM_1D/gen_chem_1D/gen_models/reinvent/train_agent.py --yaml_file input_bias_sample_iter1.yml

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Elapsed time: $execution_time seconds"
