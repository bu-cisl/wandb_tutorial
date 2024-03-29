# Installing `wandb`
This is a tutorial/template for BU researchers to integrate `wandb` in their ML stack with the [Boston University Shared Computing Cluster (SCC)](https://www.bu.edu/tech/support/research/computing-resources/scc/), the batch system of which is based on the [Sun Grid Engine](https://gridscheduler.sourceforge.net/) scheduler.

Begin by installing `wandb` in your virtual environment with 

```
pip install wandb
```

Then login to your `wandb` account using
```
wandb login
```
which will prompt you for your API key. If you don't have an API key, you may log on [here](https://wandb.ai/authorize) to retrieve it. When you paste it in the terminal, it won't show anything -- that's ok, it's for security.

# Basic `wandb`
I provided a basic sample of how to use the basic features of `wandb`, which are `wandb.log` and `wandb.watch` in `basic_train.py`. You may define the project name and entity (entity is either the `wandb` team name which is `cisl-bu`, or your `wandb` username) in the config `dict`.
Without the `entity` field, it will default to your wandb `username` and without the `project` field, it will default to "Uncategorized." 

If you wish to run a more serious/heavy experiment on an SCC compute node `run.qsub` is the qsub file you may run with 
```
qsub run.qsub
```
Modify the paths, and requested resources accordingly. `qsub` options for requesting resources and batch scripts examples may be found [here](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/).

# Hyperparameter search: `wandb.sweep`
I provided a simple template for running hyperparameter search on batch jobs on the SCC, the relevant files are `sweep.yaml`, `sweep.qsub`, `sweep.sh` and `sweep_train.py`. 

First, define your configuration parameters in `sweep.yaml`. You may then instantiate a `sweep` in the CLI with
```
wandb sweep --project <project_name> --entity <entity_name> sweep.yaml
```
which will print out the `sweep_id` you need to run the hyperparameter search. It looks like this:


`sweep.sh` is a wrapper for the qsub batch script that you will ultimately run.

