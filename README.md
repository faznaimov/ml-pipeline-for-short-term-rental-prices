# ML Pipeline for Short-Term Rental Prices in NYC

- Project **Build ML Pipeline for Short-Term Rental Prices in NYC** in [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)  program by Udacity.

## Table of Contents

- [Introduction](#ml-pipeline-for-short-term-rental-prices-in-nyc)
- [Project Description](#project-description)
- [Files and Data Description](#files-and-data-description)
- [Usage](#usage)
  * [Create Environment](#create-environment)
  * [Weights and Biases API Key](#weights-and-biases-api-key)
  * [Cookie Cutter](#cookie-cutter)
  * [Running Pipeline](#running-pipeline)
- [License](#license)

## Project Description
This project is on building a complete end to end ML pipeline to predict rental prices for airbnb rentals and make it reusable.

## Files and Data description
Building a reproducible ML pipeline will require different components which will be needed to be contained in there own environment. The following image shows the pipeline contained within weights and biases. You can check the pipeline at W&B [here](https://wandb.ai/faz-naimov/nyc_airbnb/groups/v1.0.0)

![Pipeline](/images/pipeline_graph_view.png)

The pipeline shows each component with input and output artifacts for each component.
- ```data_get```: Upload the data from local path to W&B
- ```eda```: A notebook which contains EDA for the dataset
- ```data_clean```: Clean the dataset and handle outliers
- ```data_tests```: Performs data validation
- ```data_split```: Splits the dataset to trainval and test
- ```train_random_forest```: Builds and trains a pipeline which includes handling of missing data, some feature engineering, modeling and generates scoring results.
- ```test_model```: Evaluates the saved pipeline on the test data and generates scoring results.

## Usage

### Create Environment
Make sure to have conda installed and ready, then create a new environment using the ``environment.yml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Weights and Biases API Key
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

### Cookie Cutter
You can use Cookie cutter to create stubs for new pipeline components. It is not required that you use this, but it might save you from a bit of boilerplate code. Just run the cookiecutter and enter the required information, and a new component will be created including the `conda.yml` file, the `MLproject` file as well as the script. You can then modify these as needed, instead of starting from scratch.
For example:

```bash
> cookiecutter cookie-mlflow-step -o src

step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

This will create a step called ``basic_cleaning`` under the directory ``src`` with the following structure:

```bash
> ls src/basic_cleaning/
conda.yml  MLproject  run.py
```

### Running Pipeline
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
>  mlflow run .
```
This will run the entire pipeline.

You can run one step at the time by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```
If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```

You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. You can find all adjustable features in ```congif.yaml```. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```



Below command will remove *ALL* the environments with a name starting with `mlflow`. Use at your own risk

```
> for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

## License

[License](LICENSE.txt)
