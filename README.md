# Learning Effective Hamiltonian form thermodynamic datas


we wish to propose a systematic approach of fitting thermodynamic data
which is unbiased and more efficient.


Now we have a first version demo where we show a proof-of-principle case
by learning the hamiltonian of a XXZ heisenberg chain.

![](https://github.com/yusizhuo/QMagen/blob/master/Illustration/Illust_SC.png)

```bash
jupyter notebook Demo.ipynb
```
Too run the demo, make sure you have these packages properly intstalled.

[1] Pytorch
https://pytorch.org/get-started/locally/
```bash
pip install torch torchvision torchaudio
```
[2] Bayesian Optimization
https://github.com/fmfn/BayesianOptimization
```bash
pip install bayesian-optimization
conda install -c conda-forge bayesian-optimization
```

We plan to release a first official version of this project in 
late-December, 2020.