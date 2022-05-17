from tensorflow import keras
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers


def residual_block(x, dilation, n_filters, kernel_size, l2):
    x_in = x
    x = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation, padding='causal', kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation,padding='causal',kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = x + layers.Conv1D(filters=n_filters,kernel_size=1,dilation_rate=1,kernel_regularizer=keras.regularizers.l2(l2))(x_in)
    x = layers.Activation('relu')(x)
    return x


def tcn(P):
    x_in = layers.Input(shape=(P['time_steps_in'], P['n_vars']))
    
    x = x_in
    for d in P['dilations']:
        x = residual_block(x, dilation=d, n_filters=P['n_filters'], kernel_size=P['kernel_size'], l2=P['l2'])
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(P['time_steps_out']*len(P['quantiles']), kernel_regularizer=keras.regularizers.l2(P['l2']))(x)
    out_quantiles = tf.reshape(x, (-1, P['time_steps_out'], len(P['quantiles'])))
  
    model = keras.Model(inputs=[x_in], outputs=[out_quantiles])
    # model.summary()
    
    return model


def LSTM_stateful(P):
    
    # Since we use return_sequences=True, we must specify batch shape explicitly
    x_in = layers.Input(batch_shape=(P['batch_size'], P['time_steps_in'], P['n_vars']))
    
    x = x_in
    if P['n_layers'] > 1:
        for i in range(P['n_layers']-1):
            x = layers.LSTM(P['units'],
                            stateful=True, 
                            return_sequences=True, 
                            kernel_regularizer=keras.regularizers.l2(P['l2']))(x)
            
    
    x = layers.LSTM(P['units'], 
                    stateful=True, 
                    kernel_regularizer=keras.regularizers.l2(P['l2']))(x)
    x = layers.Dense(P['time_steps_out']*len(P['quantiles']), kernel_regularizer=keras.regularizers.l2(P['l2']))(x)
    out_quantiles = tf.reshape(x, (-1, P['time_steps_out'], len(P['quantiles'])))
   
    model = keras.Model(inputs=[x_in], outputs=[out_quantiles])
    # model.summary()
    
    return model


def _pin_loss(labels, pred, quantiles):

    loss = []
    for i,q in enumerate(quantiles):
        error = tf.subtract(labels,pred[:,:,i])
        loss_q = tf.reduce_mean(tf.maximum(q*error,(q-1)*error))
        loss.append(loss_q)
    L = tf.convert_to_tensor(loss)
    total_loss = tf.reduce_mean(L)
    return total_loss


def pi_cov(y_true, y_pred):
    """ 
    Compute average coverage of prediction intervals
    """
    coverage = tf.reduce_mean(
        tf.cast((y_true >= y_pred[:,:,0])&(y_true <= y_pred[:,:,2]), tf.float32))
    return coverage


def pi_len(y_true, y_pred):
    """ 
    Compute length of prediction intervals
    """
    avg_length = tf.reduce_mean(tf.abs(y_pred[:,:,2] - y_pred[:,:,0]))
    avg_length = avg_length/(tf.reduce_max(y_true) - tf.reduce_min(y_true))
    return  avg_length


class keras_model():
    
    def __init__(self, P):
        
        self.P = P
        
        if P['model_type'] == 'lstm':
            self.model = LSTM_stateful(P)
        elif P['model_type'] == 'tcn':
            self.model = tcn(P)
        else:
            raise ValueError("model_type must be 'lstm' or 'tcn'")
            
    def fit(self, train_x, train_y, val_x, val_y, epochs=100, patience=10, verbose=0):
        
        # Create a tf Dataset. 
        tf_train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).repeat().batch(self.P['batch_size'])
        val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y)).repeat().batch(self.P['batch_size'])
        
        # Since we use repeat(), we must specify the number of times we draw a bach in an epoch
        TRAIN_STEPS = int(np.ceil(train_x.shape[0]/self.P['batch_size']))
        VAL_STEPS = int(np.ceil(val_x.shape[0]/self.P['batch_size']))
        
        
        if self.P['regression'] == 'quantile':
            self.model.compile(optimizer='adam',
                    loss=[lambda y_true, y_pred: _pin_loss(y_true, y_pred, self.P['quantiles'])],
                    metrics=[pi_cov, pi_len])
        elif self.P['regression'] == 'linear':
            self.model.compile(optimizer='adam',
                    loss='mse')
            
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
        )
        
        history = self.model.fit(tf_train_data,
                       validation_data=val_data,
                       epochs=epochs,
                       steps_per_epoch=TRAIN_STEPS,
                       validation_steps=VAL_STEPS,
                       callbacks=[es],
                       verbose=verbose)
        
        return history
    
    
    def transform(self, data_x):
        
        tf_data = tf.data.Dataset.from_tensor_slices(data_x).repeat().batch(self.P['batch_size'])
        it = iter(tf_data)
        n_steps = int(np.ceil(data_x.shape[0]/self.P['batch_size']))
        
        preds =[]
        for _ in range(n_steps):
            batch = next(it)
            preds.append(self.model(batch))
            
        preds = np.concatenate(preds, axis=0)
        preds = preds[:data_x.shape[0],:,:]
        
        return preds
    
    
class rf_model():
    def __init__(self, P):
        self.P = P
        self.model = RandomForestRegressor(n_estimators=P['n_trees'])

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        self.model.fit(train_x.reshape(train_x.shape[0],-1), train_y)
        
    def transform(self, data_x, percentile=90):
        data_x = data_x.reshape(data_x.shape[0], -1)
        prediction_int = np.zeros((data_x.shape[0], self.P['time_steps_out'], 3))
        preds = []
        for tree in self.model.estimators_:
            preds.append(tree.predict(data_x))
        preds = np.stack(preds, axis=-1)
        
        prediction_int[:,:,0] = np.percentile(preds, self.P['quantiles'][0]*100, axis=-1)
        prediction_int[:,:,1] = np.percentile(preds, self.P['quantiles'][1]*100, axis=-1)
        prediction_int[:,:,2] = np.percentile(preds, self.P['quantiles'][2]*100, axis=-1)
        
        return prediction_int
    
    
def regression_model(P):
    
    if P['model_type'] in ['lstm', 'tcn']:
        return keras_model(P)
    elif P['model_type'] == 'rf':
        return rf_model(P)
    else:
        raise ValueError("model_type must be 'lstm', 'tcn', or 'rf'")