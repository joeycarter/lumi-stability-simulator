# Luminosity Stability Simulator

Run toy experiments to simulate the "luminosity stability" problem.

Any luminosity algorithm may in principle drift over time with respect to the
"true" luminosity, in addition to having its own run-to-run statistical
fluctuations. This program simulates these luminosity algorithms under a variety
of scenarios to study how best to extract a stability uncertainty.

## Requirements

* [`numpy`](https://numpy.org/)
* [`pandas`](https://pandas.pydata.org/)
* [`ROOT`](https://root.cern/)
* [`atlasplots`](https://atlas-plots.readthedocs.io)

## Usage

```console
$ python3 stability-simulator.py [-n <number of runs>]
```
