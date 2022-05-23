import numpy as np
from sklearn.preprocessing import MinMaxScaler


class WindowGenerator():
  
    def __init__(self, input_width, label_width, shift, df, label_columns=None):
    
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
    
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
    
        self.total_window_size = input_width + shift
    
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = np.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            
        return inputs, labels
    

def data_windowing(df, B, time_steps_in, time_steps_out, shift=None, label_columns=None, train_len=.8, val_len=.1, val_data=None, test_data=None):
    
    if shift is None:
        shift=time_steps_out
    
    win = WindowGenerator(
        input_width=time_steps_in, 
        label_width=time_steps_out, 
        shift=shift,
        df=df,
        label_columns=label_columns)
    
    # Split the data, if the val and test set are not given
    if val_data is None and test_data is None:
        assert(train_len+val_len < 1)
        N = df.shape[0]
        train_data = np.array(df[:int(N*train_len)].values).astype(np.float32)
        val_data = np.array(df[int(N*train_len):int(N*(train_len+val_len))].values).astype(np.float32)
        test_data = np.array(df[int(N*(train_len+val_len)):].values).astype(np.float32)
    else:
        train_data = np.array(df.values).astype(np.float32)
        val_data = np.array(val_data.values).astype(np.float32)
        test_data = np.array(test_data.values).astype(np.float32)
        
    # Initialize scaler
    scaler = xy_scaler()
    y_index = [df.columns.get_loc(col) for col in label_columns]
    scaler.fit(train_data, y_index)
    
    # Make windows
    train_window = np.stack([ train_data[i:i+win.total_window_size] for i in range(0, train_data.shape[0] - win.total_window_size, time_steps_out)])
    train_x, train_y = win.split_window(train_window)
    train_y = train_y[:,:,0]
    
    val_window = np.stack([ val_data[i:i+win.total_window_size] for i in range(0, val_data.shape[0] - win.total_window_size, time_steps_out)])
    val_x, val_y = win.split_window(val_window)
    val_y = val_y[:,:,0]
    
    test_window = np.stack([ test_data[i:i+win.total_window_size] for i in range(0, test_data.shape[0] - win.total_window_size, time_steps_out)])
    test_x, test_y = win.split_window(test_window)
    test_y = test_y[:,:,0]

    
    # Rescale data
    train_x = scaler.transform_x(train_x)
    train_y = scaler.transform_y(train_y)
    val_x = scaler.transform_x(val_x)
    val_y = scaler.transform_y(val_y)
    test_x = scaler.transform_x(test_x)
    test_y = scaler.transform_y(test_y)
    
    # Make training batches
    batch_len = int(np.floor(train_x.shape[0]/B))
    train_data = []
    for b in range(B):
        train_data.append([train_x[b*batch_len:(b+1)*batch_len], train_y[b*batch_len:(b+1)*batch_len]])
        
    return train_data, val_x, val_y, test_x, test_y, scaler
            

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
        return data
    
    def transform_y(self, data):
        data_r = data.reshape(data.shape[0]*data.shape[1], 1)
        data_r = self.y_scaler.transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1])
        return data
        
    def inverse_transform_x(self, data):
        if len(data.shape) == 2:
            data = data[..., None]
        data_r = data.reshape(data.shape[0]*data.shape[1], -1)
        data_r = self.x_scaler.inverse_transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1], -1)
        return data
    
    def inverse_transform_y(self, data):
        data_r = data.reshape(data.shape[0]*data.shape[1], 1)
        data_r = self.y_scaler.inverse_transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1])
        return data