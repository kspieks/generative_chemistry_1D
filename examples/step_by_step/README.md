# Example step-by-step workflow applied to lipophilicty dataset
Lipophilicity is an important feature of drug molecules that affects both membrane permeability and solubility. 
This dataset contains a list of 4,200 molecules curated from ChEMBL along with their experimental octanol/water distribution coefficient (i.e., logD at pH 7.4). 
It was compiled by Wu, Z. et al. MoleculeNet: A Benchmark for Molecular Machine Learning. Chem. Sci. 2018, 9, 513−530.

## Step 0: Inspect the Data
`0_visualize_lipo_data.ipynb` provides an interactive way to visualize the data and understand the distribution of
logD, number of heavy atoms, and number of tokens needed to describe the SMILES strings.
This is helpful when deciding what settings to specify within `input.yml`.

## Build Predictive Models

### Step 1: Preprocess Data for Predictive Modeling
```
bash 1_preprocess_pred_data.sh
```
This step expects a csv file as input.
The file should contain SMILES strings in a `SMILES` column as well as columns for the regression target(s) of interest;
these predictive models will be used to bias the generator in future steps.
Here, `lipo.csv` only contains one regression target i.e., logD.

Preprocessing removes any molecules that are too big or too small or that contain undesirable elements.
These settings are defined within `input.yml`.
The user can also optionally remove stereochemsitry.
All SMILES are canonicalized.

### Step 2: Train Predictive Models
```
bash 2_train_pred_models.sh
```
Trains a predictive model using the data that was cleaned in the previous step.
Default hyperparameters are used and a random seed is specified for reproducability.
The model is evaluated using an 80:20 train:test split.
The performance on the training and testing set are saved as a csv file in case the user wants to do additional analysis.
The model is then retrained on all available data and saved as a pickle file.
The saved model can be called to score the generated molecules when sampling and/or when biasing the generator.

## Build Generative Prior

### Step 3: Proprocess Data for Generative Modeling
```
bash 3_preprocess_gen_data.sh
```
This step expects a csv file as input that contains SMILES strings in a `SMILES` column.
The same preprocessing steps are applied as in step 1.
This consistency is important so that the generative models generate SMILES in a format expected by the predictive models.

### Step 4: Train a Generative Prior
```
bash 4_train_gen_prior.sh
```
Pretrains a generative model using the cleaned SMILES from the previous step.

### Step 5: Sample from the Generative Prior
```
bash 5_sample_gen_prior.sh
```
Samples from the generative prior trained in the previous step.
Generated SMILES are automatically check for validity.
Any duplicates are removed.
All generated SMILES are scored using the specified predictive models (e.g., the logD predictor from step 2), and the resutls are saved to a csv file.

## Bias the Generative Prior
This section shows examples of how to bias the pre-trained generative model to increase the probability of
generating SMILES that have properties desired by the user.
Lastly, we can optionally apply a scaffold-constraint during the generation process.

### Step 6: Bias the Generative Prior towards One Endpoint
```
bash 6_bias_gen_prior_iter1.sh
```
This step serves as a positive control to verify that everything is working properly i.e., the generative model can be biased.
This toy example applies a unidirectional bias to maximize logD. 
The syntax within the input file to achieve this is denoted by
```
# acceptable value, worst, best. this maximizes
scale: [-1.50, -1.51, 4.50]
```
such that -1.5 is the lowest value observed in the dataset, and 4.5 is the highest observed value.
In reality, drug-like molecules often require a "sweet spot" in which logD is in a specific range.
This will be explored in the next round of biasing.

### Step 7: Sample from the Biased Model
```
bash 7_sample_gen_agent1.sh
```
Samples generated SMILES from the model fine-tuned in the previosu step.


### Step 8: Bias the Generative Prior with Multiple Constraints
```
bash 8_bias_gen_prior_iter2.sh
```
Now that we've confirmed the workflow can successfully bias the generative model, let's do a slightly 
more realistic example that biases towards a sweet spot for logD and also enforces a molecular weight constraint via
```
# acceptable low, acceptable high, min value, max value
mw: [450, 550, 300, 700]
```
which will encourage the model to generate molecules whose molecular weight is between 450 and 550.
To demonstrate an example of applying rewards or penalties for specific substructures, 
this step will also assign a small penalty for benzene rings and to hydrazothiourea, which is an example filter alert
taken from Pat Walters rd_filters repo: https://github.com/PatWalters/rd_filters/blob/master/rd_filters/data/alert_collection.csv.
Users can specify either SMARTS pattersn or SMILES strings using the following syntax in the input file:
```
substructure_matching:
    smarts:
        # small penalty for hydrazothiourea
        '[N;!R]=NC(=S)N': -0.2
    smiles:
        # small penalty for benzene
        'c1ccccc1': -0.2
```

### Step 9: Sample from the Biased Model
```
bash 9_sample_gen_agent2.sh
```
Samples generated SMILES from the model fine-tuned in the previous step.
A substructure constraint is applied during sampling:
```
scaffold_constraint: 'CCOc1cc(Nc2nc3c(cc2F)ncn3C(*)c2ccc(*)cn2)n[nH]1'
```
which tells the model to only generate around the `(*)`.
More examples of scaffold-constrained generation are shown in the example notebook
`generative_chemistry_1D/notebooks/scaffold_constrained/sample_scaffold_constrained.ipynb`
