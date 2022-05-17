import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def transform_to_windows(data_org, key_id):
    """
    Create dataframe with hours as columns 
    """
    #from the original datetime index create new columns with each of the year, month, day, and hour.
    data = data_org.copy()
    data.loc[:,'year'] = data.index.year
    data.loc[:,'month'] = data.index.month
    data.loc[:,'day'] = data.index.day
    data.loc[:,'hours'] = data.index.hour
    #construct datetimes from the split year, month, day columns
    data.loc[:,'date'] = pd.to_datetime(data.loc[:,['year', 'month', 'day']], format='%Y-%m-%d', errors='ignore')
    #set the index to dates only
    data = data.set_index(pd.DatetimeIndex(data['date']))
    #drop non target columns 
    data = data.loc[:,[key_id, 'hours']]
    #pivot the table into the format Date h0, h1, ...h23
    data = data.pivot(columns='hours', values=key_id)
    data = data.fillna(value=0) # data.dropna()
    if data.shape[0] > int(np.ceil(data_org.shape[0]/24)):
        data.drop(data.tail(1).index,inplace=True)
    return data


def split_sequences(sequences, n_steps):
    """
    Split data into observations and labels 

    source: https://github.com/nicholasjhana/short-term-energy-demand-forecasting
    """
    max_step=n_steps
    n_steps+=1
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + max_step
        #create a list with the indexes we want to include in each sample
        slices = [x for x in range(end_ix-1,end_ix-n_steps, -1)] + [y for y in range(end_ix-n_steps, i, -7)]
        #reverse the slice indexes
        slices = list(reversed(slices))
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequences[slices, :]
        seq_y = sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    X = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2],1))
    y = np.array(y)
    return X, y


def _df_to_sequence(data, target_idx=[0]):
    X_list = []
    Y_list = []
    for i,s in enumerate(data.columns): 
        x, y = split_sequences(transform_to_windows(data, key_id=s).values,7)
        if i in target_idx:
            Y_list.append(y)
        X_list.append(x)
    X = np.stack(X_list, axis=2).squeeze().astype(np.float32)
    Y = np.stack(Y_list, axis=2).squeeze().astype(np.float32)
    return X, Y


def create_datasets(data, target_idx=0, scaler=None, B=None):
    """
    Create input-output pairs from a dataframe with observations
    
    target_idx specifies the index of the variable to predict
    """

    if B is None:
        X, Y = _df_to_sequence(data, target_idx=target_idx)
        if scaler is not None:
            X = scaler.transform_x(X)
            Y = scaler.transform_y(Y)
        return X, Y
        
    else:
        ensemble_data = []
        for i in range(B):
            sb_size = int(np.floor(data.shape[0]/B))
            data_i = data[i*sb_size:i*sb_size+sb_size]
            X_i, Y_i = _df_to_sequence(data_i, target_idx=target_idx)
            if scaler is not None:
                X_i = scaler.transform_x(X_i)
                Y_i = scaler.transform_y(Y_i)
            ensemble_data.append([X_i, Y_i])
        return ensemble_data
            

class xy_scaler:
    """
    Transform X and Y data
        
    """
    
    def __init__(self, Scaler=MinMaxScaler):
        self.x_scaler = Scaler()
        self.y_scaler = Scaler()
        
    def fit(self, data, y_index):
        """
        Parameters
        ----------
        data : must have shape [time_steps, variables]
        y_index : list specifying the target variables

        """
        assert(len(data.shape) == 2)
        assert(isinstance(y_index, list))
        self.x_scaler.fit(data)
        data_y = data[:, y_index]
        self.y_scaler.fit(data_y)
                
    def transform_x(self, data):
        if len(data.shape) == 2:
            data = data[..., None]
        data_r = data.reshape(data.shape[0]*data.shape[1], -1)
        data_r = self.x_scaler.transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1], -1)
        return data.squeeze()
    
    def transform_y(self, data):
        if len(data.shape) == 2:
            data = data[..., None]
        data_r = data.reshape(data.shape[0]*data.shape[1], -1)
        data_r = self.y_scaler.transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1], -1)
        return data.squeeze()
        
    def inverse_transform_x(self, data):
        if len(data.shape) == 2:
            data = data[..., None]
        data_r = data.reshape(data.shape[0]*data.shape[1], -1)
        data_r = self.x_scaler.inverse_transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1], -1)
        return data.squeeze()
    
    def inverse_transform_y(self, data):
        if len(data.shape) == 2:
            data = data[..., None]
        data_r = data.reshape(data.shape[0]*data.shape[1], -1)
        data_r = self.y_scaler.inverse_transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1], -1)
        return data.squeeze()
             
    