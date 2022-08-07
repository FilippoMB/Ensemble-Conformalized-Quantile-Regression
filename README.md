Python implementation of the ensemble conformalized quantile regression (EnCQR) algorithm, as presented in the original [paper](https://arxiv.org/abs/2202.08756). 
EnCQR allows to generate accurate prediction intervals when predicting a time series with a generic regression algorithm for time series forecasting, such as a Recurrent Neural Network or Random Forest.

---
### Example of usage
The code in [main_EnCQR.py](https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression/blob/main/main_EnCQR.py) shows a quick example of how to perform probabilistic forecasting with EnCQR.

A detailed tutorial can be found in this [notebook](https://nbviewer.org/github/FilippoMB/Ensemble-Conformalized-Quantile-Regression/blob/main/example.ipynb), which explaines how the dataset are preprocessed and shows the differences between different regression models (LSTM, Temporal Convolutional Network, and Random Forest), which can be used as base models in the EnCQR ensemble.

----
### Citation
Consider citing the original paper if you are using EnCQR in your reasearch

	@misc{jensen2022ensemble,
	      title={Ensemble Conformalized Quantile Regression for Probabilistic Time Series Forecasting}, 
	      author={Vilde Jensen and Filippo Maria Bianchi and Stian Norman Anfinsen},
	      year={2022},
	      eprint={2202.08756},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}
	}
