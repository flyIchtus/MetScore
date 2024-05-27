# MetScore

Building a modulable, efficient score lib for the analysis of high-res weather fields from sockets.

The project is a joint effort of Météo-France GMAP/PREV research team and the CNRS PNRIA consortium (GENS project).

The intended use mode is research / experimental validation of ideas. Adjusting the functionalities to your needs is encouraged.

This code is provided with no warranty of any kind, and is under APACHE2.0 license.

Main contributors:
 - Julien Rabault (@JulienRabault), PNRIA, CNRS
 - Cyril Regan (@cyril-data), PNRIA, CNRS
 - Clément Brochet (@flyIchtus), Météo-France
 - Gabriel Moldovan (@gabrieloks), Météo-France (currently works @ ECMWF)
 
## Installation

```bash
pip install -r requirements.txt
```

## Usage
### Command line
```bash
python main.py --config config/config.yaml
```
### Input and outputs
By default, your data is supposed to be stored as .npy files. Data is either forecasts, analysis or observations, which can be indexed by both a date and lead time/validity. Samples can feature any number of weather variables.
The precise way the data is organized inside the file can be user-defined (see section on contribution strategy), but usually it is supposed to have either a "one file per sample" organization (meaning one date and lead time per sample).

## Build your config file

### Config file definition
The config file is intended to:
- define "Experiments" to be run in parallel
- define, for each Experiments, the Datasets you want to compute metrics on and the way data is presented to the metrics (Dataloaders)
- define the way you want to preprocess your data (Preprocessors)
- define the Metrics you want to compute on the given data
Each of these objects is "Configurable" (this is an ABC), meaning you can define their attributes in the config file.

## Flexible contribution strategy

The object-oriented structure of the code means you can, and are invited to, *add functionalities through subclassing and inheritance*.
The purpose is to make much of the code extendable, either to add a sampling strategy, preprocess your inputs in different ways, add innovative metrics, add new sources of observations.
Provided you have predefined the functions you want to implement, getting the whole system working should take no more than half a day work for substantial modifications.

## MetScore runtime tricks

### Uniqueness of the configuration
The desired behaviour is the code forbidding the rewrite of already computed score files. Therefore, if you run the code twice and provide the exact same configuration file, an exception will be raised. To get around:

- relocate the output folder, or rename it in the config file
- rename your Experiments, so that the code knows it is doing "something new"
- delete the previous Experiment folder
- select only metrics you have not already computed in the following config file. The experiment log file will update accordingly.

### Parallel execution

### Caching

### Detailed
