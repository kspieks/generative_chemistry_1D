# Example of Fully Automated Workflow applied to ESOL Dataset
ESOL is a small dataset consisting of experimental water solubility data for 1128 compounds from J. S. Delaney, J. Chem. Inf. Model., 2004, 44, 1000–1005.

`0_visualize_esol_data.ipynb` provides an interactive way to visualize the data and understand the distribution of
logSolubility, number of heavy atoms, and number of tokens needed to describe the SMILES strings.
This is helpful when deciding what settings to specify within `input.yml`.

`run_all_steps.py` demonstrates how the various steps from the step-by-step workflow
can be chained together to create a convenient and automated workflow. 
Use the following command to run this example:
```
bash run_fully_automated.sh
```
