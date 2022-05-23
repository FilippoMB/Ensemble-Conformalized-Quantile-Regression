import numpy as np
import matplotlib.pyplot as plt

def compute_coverage_len(y_test, y_lower, y_upper, verbose=False, eta=30, mu=0.9):
    """ 
    Compute average coverage and length of prediction intervals
    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / np.prod(y_test.shape)
    avg_length = np.mean(abs(y_upper - y_lower))
    avg_length = avg_length/(y_test.max()-y_test.min())
    cwc = (1-avg_length)*np.exp(-eta*(coverage-mu)**2)
    if verbose==True:
        print(f"PI coverage: {coverage*100:.1f}%, PI avg. length: {avg_length:.3f}, CWC: {cwc:.3f}")
    else:
        return coverage, avg_length, cwc


def asym_nonconformity(label, low, high):
    """
    Compute the asymetric conformity score
    """
    error_high = label - high 
    error_low = low - label
    return error_low, error_high


def plot_PIs(true, pred_mean, PI_low=None, PI_hi=None, 
             conf_PI_low=None, conf_PI_hi=None, 
             x_lims=None, scaler=None, title=None):
    
    if scaler:
        true = scaler.inverse_transform_y(true)
        pred_mean = scaler.inverse_transform_y(pred_mean)
    true = true.flatten()
    pred_mean = pred_mean.flatten()
    
    plt.set_cmap("tab10")
    plt.cm.tab20(0)
    plt.figure(figsize=(12, 3.5))
    plt.plot(np.arange(true.shape[0]), true, label='True', color='k')
    plt.plot(pred_mean, label='0.5', color=plt.cm.tab10(1))
    
    if conf_PI_low is not None:
        
        if scaler:
            conf_PI_low = scaler.inverse_transform_y(conf_PI_low)
            conf_PI_hi = scaler.inverse_transform_y(conf_PI_hi)
            PI_low = scaler.inverse_transform_y(PI_low)
            PI_hi = scaler.inverse_transform_y(PI_hi)
        conf_PI_hi = conf_PI_hi.flatten()
        conf_PI_low = conf_PI_low.flatten()
        PI_hi = PI_hi.flatten()
        PI_low = PI_low.flatten()    
        plt.fill_between(np.arange(true.shape[0]), conf_PI_low, conf_PI_hi, alpha=0.3, label='Conformalized')
        plt.plot(PI_low, label='original', color=plt.cm.tab10(0), linestyle='dashed')
        plt.plot(PI_hi, color=plt.cm.tab10(0), linestyle='dashed')
        
    if (conf_PI_low is None) and (PI_low is not None):
        if scaler:
            PI_low = scaler.inverse_transform_y(PI_low)
            PI_hi = scaler.inverse_transform_y(PI_hi)
            
        PI_hi = PI_hi.flatten()
        PI_low = PI_low.flatten()  
        plt.fill_between(np.arange(true.shape[0]), PI_low, PI_hi, alpha=0.3, label='PI')
        
    if x_lims is not None:
        plt.xlim(x_lims)
    plt.legend(loc='upper right')
    plt.grid()
    
    if title is not None:
        plt.title(title)
    
    plt.show()
    
    
def plot_history(history):
    
    hist_dict = history.history
    
    plt.set_cmap("tab10")
    plt.cm.tab20(0)
    fig, axs = plt.subplots(1,3, figsize=(9,2.5))
    
    
    axs[0].plot(hist_dict['loss'], label='tr_loss', color='k')
    axs[0].plot(hist_dict['val_loss'], label='val_loss', color='k', linestyle='dashed')
    axs[0].legend()
    
    axs[1].axhline(y=0.9, color='r', linestyle='-')
    axs[1].plot(hist_dict['pi_cov'], label='tr_pi_cov', color=plt.cm.tab10(0))
    axs[1].plot(hist_dict['val_pi_cov'], label='val_pi_cov', color=plt.cm.tab10(0), linestyle='dashed')
    axs[1].legend(loc='lower right')
    
    axs[2].plot(hist_dict['pi_len'], label='tr_pi_len', color=plt.cm.tab10(1))
    axs[2].plot(hist_dict['val_pi_len'], label='val_pi_len', color=plt.cm.tab10(1), linestyle='dashed')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    