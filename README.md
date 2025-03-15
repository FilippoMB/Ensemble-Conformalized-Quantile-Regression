[![arXiv](https://img.shields.io/badge/arXiv-2202.08756-b31b1b.svg?)](https://arxiv.org/abs/2202.08756)

Python implementation of the ensemble conformalized quantile regression (EnCQR) algorithm, as presented in the paper [Ensemble Conformalized Quantile Regression for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2202.08756) by V. Jensen, [F. M. Bianchi](https://sites.google.com/view/filippombianchi/home) and S. N. Anfinsen.

## TL;DR

EnCQR is a post-hoc method for uncertainty quantification. It creates valid prediction intervals on top of a generic regression algorithm for time series forecastings, such as a Recurrent Neural Network, ARIMA, Random Forest, and so on.

## Example of usage
The code in [main_EnCQR.py](https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression/blob/main/main_EnCQR.py) shows a quick example of how to perform probabilistic forecasting with EnCQR.

A detailed tutorial can be found in this [![nbviewer](https://img.shields.io/badge/-notebook-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.org/github/FilippoMB/Ensemble-Conformalized-Quantile-Regression/blob/main/example.ipynb), which explaines how the datasets are preprocessed and it shows the differences when using different regression models (LSTM, Temporal Convolutional Network, and Random Forest) as base models in the EnCQR ensemble.

## Citation
Please, consider citing the original paper if you use EnCQR in your research.

```bibtex
@article{jensen2022ensemble,
  title={Ensemble conformalized quantile regression for probabilistic time series forecasting},
  author={Jensen, Vilde and Bianchi, Filippo Maria and Anfinsen, Stian Normann},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```
