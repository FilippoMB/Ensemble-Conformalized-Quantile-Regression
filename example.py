import pandas as pd
from datetime import timedelta

import data_preprocessing
from models import regression_model
from conformal_prediction import EnCQR
import utils


############### Hyperparameters
target_idx = [0]            # target variables to predict
B = 3                       # number of ensembles
alpha = 0.1                 # confidence level            
quantiles = [alpha/2,       # quantiles to predict
             0.5,
             1-(alpha/2)] 
model_type = 'tcn'          # options: {'rf', 'lstm', 'tcn'}

# rf only
n_trees = 20                # number of trees in each rf model

# lstm and tcn only
regression = 'quantile'     # options: {'quantile', 'linear'}. If 'linear', just set one quantile
l2_lambda = 1e-4            # weight of l2 regularization in the lstm and tcn models
batch_size = 16             # size of batches using to train the lstm and tcn models

# lstm only
units = 128                 # number of units in each lstm layer
n_layers = 3                # number of lstm layers in the model

# tcn only
dilations = [1,2,4,8]       # dilation rate of the Conv1D layers
n_filters = 128             # filters in each Conv1D layer 
kernel_size = 7             # kernel size in each ConvID layer

# Store the configuration in a dictionary
P = {'B':B, 'alpha':alpha, 'quantiles':quantiles, 'model_type':model_type,
     'n_trees':n_trees,  
     'regression':regression,'l2':l2_lambda, 'batch_size':batch_size,
     'units':units,'n_layers':n_layers,
     'dilations':dilations, 'n_filters':n_filters, 'kernel_size':kernel_size}
##############################

############### Load data
url = 'https://raw.githubusercontent.com/Duvey314/austin-green-energy-predictor/master/Resources/Output/Webberville_Solar_2017-2020_MWH.csv'
df = pd.read_csv(url)
df = df.drop(columns=('Weather_Description'))
df = df.drop(columns=('Year'))
df = df.drop(columns=('Month'))
df = df.drop(columns=('Day'))
df = df.drop(columns=('Hour'))
df = df.drop(columns=('Date_Time'))

# create date+hour index 
date_list = pd.date_range(start='01/01/2017', end='31/07/2020')
date_list = pd.to_datetime(date_list)
hour_list = []
for nDate in date_list:
    for nHour in range(24):
        tmp_timestamp = nDate+timedelta(hours=nHour)
        hour_list.append(tmp_timestamp)
date_list = pd.to_datetime(hour_list) 
df['hour_list'] = date_list[:-1]
df = df.set_index('hour_list')

# train, val, test datasets
df_train = df[0:365*24]
df_val = df[365*24:365*24*2]
df_test = df[365*24*2:365*24*3]
##############################

############### Data preprocessing
# Initialize the scaler
Scaler = data_preprocessing.xy_scaler()
Scaler.fit(df_train.values, target_idx)

# split into input-output pairs
train_data = data_preprocessing.create_datasets(df_train, target_idx=target_idx, scaler=Scaler, B=B) 
val_x, val_y = data_preprocessing.create_datasets(df_val, target_idx=target_idx, scaler=Scaler)
test_x, test_y = data_preprocessing.create_datasets(df_test, target_idx=target_idx, scaler=Scaler)
print("-- Training data --")
for i in range(len(train_data)):
    print(f"Set {i} - x: {train_data[i][0].shape}, y: {train_data[i][1].shape}")
print("-- Validation data --")
print(f"x: {val_x.shape}, y: {val_y.shape}")
print("-- Test data --")
print(f"x: {val_x.shape}, y: {val_y.shape}")

# Update configuration dict
P['time_steps_in'] = test_x.shape[1]
P['n_vars'] = test_x.shape[2] 
P['time_steps_out'] = test_y.shape[1]
##############################


############## Train and test quantile regression models
# Train 
model = regression_model(P)
hist = model.fit(train_data[0][0], train_data[0][1], val_x, val_y)
utils.plot_history(hist)

# Test
PI = model.transform(test_x)
utils.plot_PIs(test_y, PI[:,:,1],
                PI[:,:,0], PI[:,:,2],
                x_lims=[1000,1168], scaler=Scaler, title='TCN model')

# LSTM model
P['model_type'] = 'lstm'
model = regression_model(P)
hist = model.fit(train_data[0][0], train_data[0][1], val_x, val_y)
utils.plot_history(hist)


PI = model.transform(test_x)
utils.plot_PIs(test_y, PI[:,:,1],
                PI[:,:,0], PI[:,:,2],
                x_lims=[1000,1168], scaler=Scaler, title='LSTM model')


# rf model
P['model_type'] = 'rf'
model = regression_model(P)
model.fit(train_data[0][0], train_data[0][1])

PI = model.transform(test_x)
utils.plot_PIs(test_y, PI[:,:,1],
                PI[:,:,0], PI[:,:,2],
                x_lims=[1000,1168], scaler=Scaler, title='RF model')
##############################


############ Conformalization
P['model_type'] = 'tcn'
PI, conf_PI = EnCQR(train_data, val_x, val_y, test_x, test_y, P)

utils.plot_PIs(test_y, PI[:,:,1],
               PI[:,:,0], PI[:,:,2],
               conf_PI[:,:,0], conf_PI[:,:,2],
               x_lims=[1000,1168], scaler=Scaler)

print("Quantile regression:")
utils.compute_coverage_len(test_y.flatten(), PI[:,:,0].flatten(), PI[:,:,2].flatten(), verbose=True)
print("Conformalized quantile regression:")
utils.compute_coverage_len(test_y.flatten(), conf_PI[:,:,0].flatten(), conf_PI[:,:,2].flatten(), verbose=True)
#############################