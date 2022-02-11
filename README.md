Python implementation of the ensemble conformalized quantile regression (EnCQR) algorithm, as presented in the original [paper](https://arxiv.org/). 
EnCQR allows to generate accurate prediction intervals when predicting a time series with a generic regression algorithm for time series forecasting, such as a Recurrent Neural Network or Random Forest.

---
### Examples

The notebooks show how to use EnCQR on top of two popular regression algorithms:
- **Random Forest**: you can access the notebook [here](https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression-forProbabilistic-Time-Series-Forecasting/blob/main/EnCQR_with_RF.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12eqEZoXEnZ1G207bbuxD2X6c86FY_iiz?usp=sharing). The notebook also computes prediction intervals with a standard Quantile Regression and with the [EnBPI method](http://proceedings.mlr.press/v139/xu21h.html?ref=https://codemonkey.link).
- **LSTM** you can access the notebook [here](https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression-forProbabilistic-Time-Series-Forecasting/blob/main/EnCQR_with_LSTM.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19MFZ8KVe9s9Rs505u9kxJp4rmcFDqMU-?usp=sharing).

----
### Citation
Consider citing the original paper if you are using EnCQR in your reasearch

	@article{vjensen2021encqr,
	  title={Ensemble Conformalized Quantile Regression for Probabilistic Time Series Forecasting},
	  author={Vilde, Jensen and Bianchi, Filippo Maria and Anfinsen, Stian},
	  journal={},
	  year={2021},
	  publisher={}
	}
