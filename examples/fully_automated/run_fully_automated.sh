#!/bin/bash

start_time=$(date +%s)

GEN_CHEM_1D=../../../generative_chemistry_1D

# confirm that the correct environment is being used
which python

python run_all_steps.py --yaml_file input.yml

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Elapsed time: $execution_time seconds"
