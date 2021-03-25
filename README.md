# PyQMagen - a Python package for thermal data analysis of quantum magnets

The Python version of QMagen provides a highly 
customizable package for the analysis of thermal data 
of quantum magnets. 
Featuring Bayesian optimizer for the fitting loss, 
and combined with an ED solver,
the PyQMagen is a computational light package 
that can analyze high temperature thermal data.

## Installation

Firstly, please `cd` to your customized path
```shell script
cd user_customized_path
```

The QMagen package can then be installed 
locally via following commands:

```shell script
git clone https://github.com/QMagen/PyQMagen.git
cd PyQMagen
```
### Dependencies

If you are new to Python, we strongly recommended to use 
*anaconda* or *miniconda* to configure your environment.
They can be found
[here](https://www.anaconda.com/).

After installing conda, you can create and activate a separate 
virtual environment via 

```shell script
conda create -n qmagen
```

Then you can install the PyQMagen package and its dependencies
into the ```qmagen``` environment with 
```shell script
conda activate qmagen
pip install -e .
```

> if you don't have pip installed by default, run  
>```conda install pip``` or 
>```conda install python``` 
> should fix the issue
---
### Demo
After installation, the PyQMagen package can be imported in 
Python environment via

```python
import qmagen
```

for example, let's see how to calculate the simulated
thermal data of a uniform Heisenberg spin-chain of 8
spins

```python
import numpy as np
from qmagen import solver
from qmagen.models import chain

# create a spin-chain model with 8 spins
mymodel = chain.UniformSpinChain(l=8)

# get the interactions with coupling strength J=1
interactions = mymodel.generate_interactions(J=1)

# create a ED solver
mysolver = solver.EDSolver(size=mymodel.l) 

# calculate the thermal data with ED solver with generated interactions
thermal_data = mysolver.forward(interactions, T=np.linspace(0.1, 10, 100))
```


---
### Tutorial
For more detailed usage guide,
we provide a tutorial in jupyter notebook

You can run the tutorial by
```shell script
# if you don't have jupyter installed
conda install jupyter notebook

jupyter-notebook tutorial/introduction.ipynb
```

---

### Future updates

In the short future, following features will be updated:

- Large-scale 1D solver - LTRG
- Animation for the optimization process
- More templates for spin-models

Also we are working hard to bring even more 
exciting features in PyQMagen, including

- Large-scale 2D solver - XTRG
- Neural network based thermal analysis

---
### Maintainer 

- Sizhuo YU, Beihang university\
mail: [yusizhuo@buaa.edu.cn](yusizhuo@buaa.edu.cn)

- Bin-bin Chen, Beihang university\
mail: [bunbun@buaa.edu.cn](bunbun@buaa.edu.cn)


---
### Cite us

```bib
@article{QMagen2020,
  title={Learning Effective Spin Hamiltonian of Quantum Magnet},
  author={Sizhuo Yu, Yuan Gao, Bin-Bin Chen and Wei Li},
  journal={arXiv preprint arXiv:2011.12282},
  year={2020}
}
```