# PyQMagen - a Python package for thermal data analysis of quantum magnets

The Python version of QMagen provides a highly 
customizable package for the analysis of thermal data. 
Featuring Bayesian optimizer for the fitting loss, 
and combined with an ED solver,
the PyQMagen is a computational light package 
that can analyze high temperature thermal data.

## Installation

If the user is new to Python, then it is strongly recommended to use 
*anaconda* or *miniconda* to configure your environment.
They can be found
[here](https://www.anaconda.com/).

After installing conda, you can create a separate environment in
the terminal via 

```shell script
conda env -n qmagen
```
then activate the new environment by
```shell script
conda activate qmagen
```
Also it is recommended to install jupyter notebook
in order to run the tutorial
```shell script
conda install jupyter notebook
```

The QMagen package can then be installed 
locally via following commands:

```shell script
git clone https://github.com/QMagen/HamiltonianLearning.git
cd HamiltonianLearning
pip install -e .
```

We also provide a tutorial in jupyter notebook
```shell script
jupyter-notebook DEMO.ipynb
```

## Guide

### Define a model


```python
from magen.models import chain

mymodel = chain.UniformSpinChain(l=8)
interactions = mymodel.generate_interactions(J=1)
```
### Calculate simulated thermal data with many-body solvers

```python
import magen.solver as solver

mysolver = solver.EDSolver(size=mymodel.l)
```

### Use optimizer to infer optimal model parameters
```python

```