Python implementation of the ensemble conformalized quantile regression (EnCQR) algorithm, as presented in the original [paper](https://arxiv.org/abs/2202.08756). 
EnCQR allows to generate accurate prediction intervals when predicting a time series with a generic regression algorithm for time series forecasting, such as a Recurrent Neural Network or Random Forest.

---
### Example of usage
This [notebook](https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression/blob/main/example.ipynb) shows an example of how to generate PIs with EnCQR using an LSTM, a Temporal Convolutional Network, or Random Forest regression as the base models.

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
