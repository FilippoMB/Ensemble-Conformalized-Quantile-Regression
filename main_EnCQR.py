import data_preprocessing
from conformal_prediction import EnCQR
import utils
import data_loaders


############### Hyperparameters
B = 3                       # size of the ensemble
alpha = 0.1                 # confidence level            
quantiles = [alpha/2,       # quantiles to predict
             0.5,
             1-(alpha/2)] 
model_type = 'rf'           # use Random Forest as regression model
n_trees = 10                # number of trees in each rf model

# Store the configuration in a dictionary
P = {'B':B, 'alpha':alpha, 'quantiles':quantiles, 'model_type':model_type,'n_trees':n_trees }


#################### Load data
df = data_loaders.get_met_data()
# Predict the temperature for the next 6h, given the last 24h of weather measurements
train_data, val_x, val_y, test_x, test_y, Scaler = data_preprocessing.data_windowing(df=df, 
                                                                                     B=B, 
                                                                                     time_steps_in=24, 
                                                                                     time_steps_out=6,  
                                                                                     label_columns=['T (degC)'])
# Update configuration dict
P['time_steps_in'] = test_x.shape[1]
P['n_vars'] = test_x.shape[2] 
P['time_steps_out'] = test_y.shape[1]


############ Compute PI with EnCQR
PI, conf_PI = EnCQR(train_data, val_x, val_y, test_x, test_y, P) # returns both non-conformalized and conformalized PIs

utils.plot_PIs(test_y, PI[:,:,1],
               PI[:,:,0], PI[:,:,2],
               conf_PI[:,:,0], conf_PI[:,:,2],
               x_lims=[200,300], 
               scaler=Scaler)
utils.compute_coverage_len(test_y.flatten(), conf_PI[:,:,0].flatten(), conf_PI[:,:,2].flatten(), verbose=True)