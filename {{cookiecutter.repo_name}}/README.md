# Code for training models in *nussl*

This repository contains the code for training many of the models
in *nussl*. It contains a few extra utilities, such as a simple way
to schedule jobs with GPUs and recipes for creating datasets.

## Setting up the environment

```
conda env create -f conda.yml
conda activate {{cookiecutter.repo_name}}
```

Set the paths appropriately in `env.sh`. Please ensure all paths in `env.sh` are 
absolute paths. Then `source env.sh`.

If you update the `conda.yml` file to say, add new requirements, update the conda 
environment after activating it:

```
conda env update -f conda.yml --prune
```

Note that the `--prune` option will get rid of any packages that were removed in `conda.yml`.

## Running experiments

All of the experiments in this repository use 
[gin-config](https://github.com/google/gin-config), which is a flexible
way to specify hyperparameters and configuration in a hierarchical and
reusable fashion. There are 5 main functions whose arguments are all
configured with gin:

1. `instantiate`: This function instantiates an experiment by compiling a
   gin config file on the fly, possibly sweeping across some hyperparameters,
   and writing it all to an output directory. This *must* be called first.
2. `train`: This function will train a model. This needs datasets 
   (see `wham/wham8k.gin` for an example), and model and training 
   parameters (see `wham/exp/chimera.gin` for an example).
3. `cache`: This function takes the datasets and caches them to the 
   desired location.
4. `evaluate`: This function evaluates a separation algorithm on the 
   test dataset.
5. `analyze`: This analyzes the experiment after evaluation. Spits out a report
   card with the metrics.

Finally `all` will run `train`, then `evaluate`, then `analyze`.

### Usage

The `main.py` script takes a few arguments. It has the following signature:

```
usage: main.py [-h] [-c CONFIG [CONFIG ...]] [-o OUTPUT_FOLDER] func

positional arguments:
  func                  What function to run, given the configuration. Choices
                        are train, evaluate, analyze, all, cache, instantiate,
                        resume, debug.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG [CONFIG ...], --config CONFIG [CONFIG ...]
                        List of .gin files containing bindings for relevant
                        functions.
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        If using instantiate command, need an output folder to
                        put the compiled .gin config.
```

The list of config files might first start with an
environment config file which sets all the 
machine-specific variables like path to data directories and so on, a 
data config file which describes all of the datasets, 
and an experiment config file which describes
the model settings, the optimizer, the algorithm settings, and whatever else
is needed. Generally, you'll want to follow a process like this:

```
python main.py -c [path/to/data.gin] [path/to/env.gin] [path/to/exp.gin] cache
```

This will cache all of the datasets so that things train faster. Then instantiate
a run of this combination of gin files:

```
python main.py instantiate -c [path/to/data.gin] [path/to/env.gin] [path/to/exp.gin] -o wham/exp/out/dpcl
```

This will instantiate your experiments into a single config file and place that config
into a folder. The location of the folder depends on the location of the path to the
experiment config. If the config file, for example, was at `wham/exp/dpcl.gin`, then
the output will be at `wham/exp/out/dpcl/run0:[descriptive_string]/config.gin`. If you
had a sweep in your experiment config file, then you'll see multiple of these. For
example, if you had lines in your experiment config like this:

```
sweep.parameters = {
    'add_clip_gradient_handler.clip_value': [1e-4, 1e-3, 1e-2]
}
```

Then, the output configs will be at:

```
wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin 
wham/exp/out/dpcl/run1:clip_value:0.001/config.gin 
wham/exp/out/dpcl/run2:clip_value:0.01/config.gin 
```

Now, you're ready to run experiments. To run, say the first experiment you would do:

```
python main.py -c wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin train
python main.py -c wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin evaluate
python main.py -c wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin analyze
```

Or to do all of them in one fell swoop:

```
python main.py -c wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin all
```

### Allocating a GPU and job management

You might want to run a few experiments at once, waiting for a GPU to become 
available and then launching the experiment appropriately. To do this, you
can use the allocate script, as follows:

```
./allocate.py 1 python main.py -c wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin all
```

This will allocate 1 GPU for the script, and then run the experiment. If no GPUs are
available, it will wait till one does become available and then run the experiment.

Running and tracking multiple jobs is really easy if you install 
[task-spooler](http://manpages.ubuntu.com/manpages/xenial/man1/tsp.1.html). To say,
queue up 3 jobs as above, you can do this:

```
tsp ./allocate.py 1 python main.py -c wham/exp/out/dpcl/run0:clip_value:0.0001/config.gin all
tsp ./allocate.py 1 python main.py -c wham/exp/out/dpcl/run1:clip_value:0.001/config.gin all
tsp ./allocate.py 1 python main.py -c wham/exp/out/dpcl/run2:clip_value:0.01/config.gin all
```

Run `tsp` to see everything running.

To see the logs of one of the jobs, do `tsp -t [id]`. 
By default, tsp runs the jobs one at a time. To run 3 jobs at a time, do `tsp -S 3`.
`tsp -S N` will run `N` jobs at a time.
Finally, you can easily track all of the logs by running the logs script 
(you'll need to be in a running tmux session to do this):

```
./logs.py | sh
```

## Starter code

To get started with music-based experiments, see [musdb](musdb/README.md). To get 
started with speech-based experiments, see [wham](wham/README.md).