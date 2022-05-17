import numpy as np
from models import regression_model
import utils


def EnCQR(train_data, val_x, val_y, test_x, test_y, P):
    """
    Parameters
    ----------
    train_data : list of data to train an ensemble of models
    test_x : input test data
    test_y : output test data
    P : dictionary of parameters

    Returns
    -------
    PI : original PI produced by the ensemble model
    conf_PI : PI after the conformalization
    """

    index = np.arange(P['B'])
    s = P['time_steps_out']
    
    # dict containing LOO predictions
    dct_lo = {}
    dct_hi = {}
    for key in index:
      dct_lo['pred_%s' % key] = []
      dct_hi['pred_%s' % key] = []
    
    # training a model for each sub set Sb
    ensemble_models = []
    for b in range(P['B']):
        f_hat_b = regression_model(P)
        f_hat_b.fit(train_data[index[b]][0], train_data[index[b]][1], val_x, val_y)
        ensemble_models.append(f_hat_b)
        
        # Leave-one-out predictions for each Sb
        indx_LOO = index[np.arange(len(index))!=b]
        for i in range(len(indx_LOO)):
            pred = f_hat_b.transform(train_data[indx_LOO[i]][0])
            dct_lo['pred_%s' %indx_LOO[i]].append(pred[:,:,0])
            dct_hi['pred_%s' %indx_LOO[i]].append(pred[:,:,2])
            
    f_hat_b_agg_low  = np.zeros((train_data[index[0]][0].shape[0], P['time_steps_out'], P['B']))
    f_hat_b_agg_high = np.zeros((train_data[index[0]][0].shape[0], P['time_steps_out'], P['B']))
    for b in range(P['B']):
        f_hat_b_agg_low[:,:,b] = np.mean(dct_lo['pred_%s' %b],axis=0) 
        f_hat_b_agg_high[:,:,b] = np.mean(dct_hi['pred_%s' %b],axis=0)  
        
    # residuals on the training data
    epsilon = []
    epsilon_hi=[]
    for b in range(P['B']):
        e_low,e_high = utils.asym_nonconformity(label=train_data[b][1], 
                                                  low=f_hat_b_agg_low[:,:,b], 
                                                  high=f_hat_b_agg_high[:,:,b])
        epsilon.append(e_low)
        epsilon_hi.append(e_high)
    epsilon = np.array(epsilon).flatten()
    epsilon_hi = np.array(epsilon_hi).flatten()
    
    # Construct PIs for test data
    f_hat_t_batch = np.zeros((test_y.shape[0], test_y.shape[1], 3, P['B']))
    for b, model_b in enumerate(ensemble_models):
        f_hat_t_batch[:,:,:,b] = model_b.transform(test_x)
    PI  = np.mean(f_hat_t_batch,  axis=-1) 
    
    # Conformalize prediction intervals on the test data
    conf_PI = np.zeros((test_y.shape[0], test_y.shape[1], 3))
    conf_PI[:,:,1] = PI[:,:,1]
    for i in range(test_y.shape[0]):   
    
        e_quantile_lo = np.quantile(epsilon, 1-P['alpha']/2)
        e_quantile_hi = np.quantile(epsilon_hi, 1-P['alpha']/2)
        conf_PI[i,:,0] = PI[i,:,0] - e_quantile_lo
        conf_PI[i,:,2] = PI[i,:,2] + e_quantile_hi
    
        # update epsilon with the last s steps
        e_lo, e_hi = utils.asym_nonconformity(label=test_y[i,:],
                                                low=PI[i,:,0],
                                                high=PI[i,:,2])
        epsilon = np.delete(epsilon,slice(0,s,1))
        epsilon_hi = np.delete(epsilon_hi,slice(0,s,1))
        epsilon = np.append(epsilon, e_lo)
        epsilon_hi = np.append(epsilon_hi, e_hi)
        
    return PI, conf_PI