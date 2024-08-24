#!/bin/bash

start_time=$(date +%s)

GEN_CHEM_1D=../../../generative_chemistry_1D

which python

python $GEN_CHEM_1D/gen_chem_1D/pred_models/preprocess_pred_data.py --yaml_file input.yml

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Elapsed time: $execution_time seconds"
