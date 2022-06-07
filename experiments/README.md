# Overview of Experiments

This `experiments` directory contains clean code implementations of the experiments from [[1]](https://arxiv.org/abs/2206.04843). The scripts in this directory assume that `torchlaplace` is installed following instructions from the main directory, and the associated additional packages are installed such that the baselines compared against in [[1]](https://arxiv.org/abs/2206.04843) can be used. To install the additional packages run `pip install torchlaplace[all]`.

For simple examples on how to use `torchlaplace` see the [`examples`](../examples) directory

## Demo

The [`exp_all_baselines.py`](./exp_all_baselines.py) file contains a short implementation comparing against all baselines in [[1]](https://arxiv.org/abs/2206.04843), returning a pandas DataFrame of the test RMSE extrapolation error with std across input seed runs, which is printed out to the console and the log file at the end of all the seed runs. It also saves out training meta-data in a local `./results` folder (such as training loss array and NFE array against the epochs array). The code for evaluating all baselines in this file is within a function `experiment_with_all_baselines`, that returns the pandas DataFrame of the test RMSE extrapolation error with std, to allow users to use this in their workflow or in a Jupyter notebook.

To run a single experiment run

```
python exp_all_baselines.py
```

To specify a particular dataset run

```
python exp_all_baselines.py -d lotka_volterra_system_with_delay
```

Furthermore there are allot more additional experimental parameters that can be used, for full details see the [`exp_all_baselines.py`](./exp_all_baselines.py) file.
