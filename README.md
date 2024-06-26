# MetScore

Building a modulable, efficient score lib for the analysis of high-res, limited area model weather fields, devoted to compare physics-based NWP models and data-driven emulators. A strong focus is given on metrics applying to ensembles, but deterministic systems can also be evaluated.

The project is a joint effort of Météo-France GMAP/PREV research team and the CNRS PNRIA consortium (GENS project).

The intended use mode is research / experimental validation of ideas. Adjusting the functionalities to your needs is encouraged. 

This code is provided with no warranty of any kind, and is under APACHE2.0 license.

Main contributors to the initial version:
 - Julien Rabault (@JulienRabault), PNRIA, CNRS
 - Clément Brochet (@flyIchtus), GMAP/PREV, Météo-France
 - Gabriel Moldovan (@gabrieloks), GMAP/PREV Météo-France (currently works @ ECMWF)
 
## Installation

```bash
pip install -r requirements.txt
```
This requires at least Python 3.9 to run.

## Code structure

| Path | Description |
| --- | --- |
|MetScore|Root folder of the repository|
|&ensp;&ensp;&boxvr;&nbsp; config | Where to store config files |
|&ensp;&ensp;&boxvr;&nbsp; core |Core configurable / dataset / dataloading logic|
|&ensp;&ensp;&boxvr;&nbsp; metrics | Individual implementation of metrics functions and main catalogue of metrics|
|&ensp;&ensp;&boxvr;&nbsp; preprocess | Classes and catalogue of preprocessors|
|&ensp;&ensp;&boxvr;&nbsp; stats | Functions to make statistical analysis (significance) |
|&ensp;&ensp;&boxvr;&nbsp; main.py | Executable script reading a config file and launching an experiment |
|&ensp;&ensp;&boxvr;&nbsp; plotting_autom.py | Executable script plotting score data for several score experiments |
|&ensp;&ensp;&boxvr;&nbsp; plotting_functions.py | Functions to plot for each metric where a "standard way" of plotting can be defined. |
|&ensp;&ensp;&boxvr;&nbsp; diagclass.puml | Visualisation scheme for the class diagram of the whole code. |
## Usage
### Command line
```bash
python main.py --config config/config.yaml
```
### Input and outputs
By default, your data is supposed to be stored as .npy files. Data is either made of forecasts, analysis or observations, which can be indexed by both a date and lead time/validity. Samples can feature any number of weather variables, with the common "channel-first" convention. 
The precise way the data is organized inside the file can be user-defined (see section on contribution strategy), but usually it is supposed to have either a "one file per sample" organization (meaning one date and lead time, possibly one ensemble member per sample), or "one file per ensemble" (gathering N members in a single file).

It is also mandatory that you have for *real datasets* (see meaning below) a structured .csv file with named columns, as presented in *./MetScore/config/example_labels.csv*

There is one output per metric, computed on the dataset provided, stored in .npy format.
The precise shape of the output depends on the chosen metric.

## Useful notions

### Real, fake, observations
The code expects the existence of 3 sources of data (Datasets objects):

- __real__ designates the outputs of a physics-based NWP system, serving as a baseline or reference
- __fake__ designates the outputs of a data-driven emulator "faking" the NWP system (typically as expected in a GAN).
- __observations__ designates "ground truth", against which both datasets above are evaluated

Therefore it is necessary to provide all 3 objects in the config file. 
However, depending on the chosen metrics, the corresponding data may not be used during scoring computation.

### Metrics : Batched and non-batched, distance and standalone

__Batched metrics__ apply to fixed, given situations, e.g. identified by a date/leadtime timestamp.
Therefore they apply to batches of the full dataset, hence their name. 
CRPS is a good example, as it compares one ensemble to one observation on a given date/leadtime.

__Non-batched metrics__ apply to "global" datasets, seen as "climatological" distributions. The ordering of the data samples or batches does not count.
Global bias, power spectra, wasserstein distances are such metrics. These metrics usually are not computed "per-timestamp".

__Standalone metrics__ require a single source of data to be computed, they do not *compare* two sources of data.

__Distance metrics__ are just the opposite.
E.g : power spectra or ensemble variance are standalone metrics, wasserstein distances or CRPS are distance metrics; CRPS is a distance metric.

As you can spot in the above examples, the "batched/non-batched" and "distance/standalone" distinctions are uncorrelated.

## Build your config file

### Config file definition
The config file is intended to:
- define "Experiments" to be run in parallel
- define, for each Experiments, the Datasets you want to compute metrics on and the way data is presented to the metrics (Dataloaders)
- define the way you want to preprocess your data (Preprocessors)
- define the Metrics you want to compute on the given data
Each of these objects is "Configurable" (this is an ABC), meaning you can define their attributes in the config file.

### Main configuration fields

see the comments in the example config file (./MetScore/config/config_example.yml).

## MetScore runtime tricks

### Uniqueness of the configuration
The desired behaviour is the code forbidding the rewrite of already computed score files. Therefore, if you run the code twice and provide the exact same configuration file, an exception will be raised. To get around:

- relocate the output folder, or rename it in the config file
- rename your Experiments, so that the code knows it is doing "something new"
- delete the previous Experiment folder
- select only metrics you have not already computed in the following config file. The experiment log file will update accordingly.

### Caching

If non-batched (~global) metrics are computed, these computations takes place *after* the computation for batched metrics, so that the data loaded to compute the latter is not uselessly reloaded. To that end, a cache stores any pre-loaded data to re-use them in non-batched metrics computation. If no non-batched metric is selected, the cache does not activate, leading to a reduced memory consumption.

### Parallel execution
The "unit" for parallel computing is the Experiment object. If you specify several Experiments in the same config, one process will be started per experiment.
Experiments refer to one datum of *Dataloader/Dataset[s]/Preprocessor[s]/Metric[s]* in the config file.
Starting many experiments in parallel is a reasonable efficient strategy, but keep in mind the compute/memory costs for each experiments add up, and that the caching mechanism can lead to memory leaks on large datasets if too many experiments are launched concurrently.

## Flexible contribution strategy

The object-oriented structure of the code means you can, and are invited to, __add functionalities through subclassing and inheritance__.
The purpose is to make much of the code extendable, either to add a sampling strategy, preprocess your inputs in different ways, add innovative metrics, add new sources of observations.
Provided you have predefined the functions you want to implement, getting the whole system working should take no more than half a day work for the most substantial modifications.

### Adding a new metric

### Adding a new preprocessor

### Adding a new dataloading / dataset process


## Detailed view of the different abstract classes and their implementation

### Todo


## Report issues

--> The ISSUES section is made for you
For specific questions, please FIRST file an issue, THEN write to clement dot brochet at meteo dot fr.
Any mail unseconded by an issue will be discarded.
