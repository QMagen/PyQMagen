# PyQMagen - a Python package for thermal data analysis of quantum magnets

The Python version of QMagen provides a highly 
customizable package for the analysis of thermal data 
of quantum magnets. 
Featuring Bayesian optimizer for the fitting loss, 
and combined with an ED solver,
the PyQMagen is a computational light package 
that can analyze high temperature thermal data.

## Installation

If the user is new to Python, we strongly recommended to use 
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

After installation, the PyQMagen package can be imported in 
Python environment via

```python
import magen
```
---
### Tutorial
For more detailed usage guide,
we provide a tutorial in jupyter notebook

```shell script
jupyter-notebook tutorial/introduction.ipynb
```
