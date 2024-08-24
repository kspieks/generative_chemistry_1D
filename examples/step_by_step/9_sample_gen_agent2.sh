#!/bin/bash

start_time=$(date +%s)

GEN_CHEM_1D=../../../generative_chemistry_1D

which python

python $GEN_CHEM_1D/gen_chem_1D/gen_models/reinvent/sample.py --yaml_file input_bias_sample_iter2.yml

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Elapsed time: $execution_time seconds"
