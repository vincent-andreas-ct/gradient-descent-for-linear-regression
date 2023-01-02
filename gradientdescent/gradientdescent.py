# import dependencies
import numpy as np
import math

def construct_mat(data, target, w):
    return np.matrix(data), np.matrix(target), np.matrix(w)

def calc_cost(data, target, w, b):
    """
    Calculate the cost J(w,b) of data
    Args:
    data (ndarray(m,n) or (m,)) : feature(s) or variable(s)
    target (ndarray(m,))        : target variable
    w (ndarray(n,) or scalar)   : coefficient(s) - model parameter(s)
    b (scalar)                  : intercept - model parameter

    Returns:
    cost (scalar)               : cost J(w,b) from the given w,b of data
    """
    # construct matrix
    data_m, target_m, w_m = construct_mat(data, target, w)
    try:
        # for 2-D data array
        m, n = data.shape
        pred = (data_m * w_m.T) + b
    except Exception as e:
        # for 1-D data array
        m = data.shape[0]
        pred = (data_m.T * w_m) + b
    # calculate cost J(w,b)
    diff = pred-target_m.T
    diff_s = np.array(diff)**2
    cost = np.sum(diff_s)/(2*m)
    return cost

def calc_gradient(data, target, w, b):
    """
    Calculate the gradient of the parameters w,b from the given data
    Args:
    data (ndarray(m,n) or (m,)) : feature(s) or variable(s)
    target (ndarray(m,))        : target variable
    w (ndarray(n,) or scalar)   : coefficient(s) - model parameter(s)
    b (scalar)                  : intercept - model parameter

    Returns:
    g_coeffs (ndarray(n,) or scalar)    : gradient of the parameter(s) w
    g_intercept (scalar)                : gradient of the parameter b
    """
    # construct matrix
    data_m, target_m, w_m = construct_mat(data, target, w)
    try:
        # for 2-D data array
        m, n = data.shape
        pred = (data_m * w_m.T) + b
        g_coeffs = np.zeros((n,))
        diff = pred-target_m.T
        for col in range(n):
            g_col = diff.T * data_m[:, col]
            g_coeffs[col] = g_col.item(0)/m
        g_intercept = np.sum(diff)/m
    except Exception as e:
        # for 1-D data array
        m = data.shape[0]
        pred = (data_m.T * w_m) + b
        diff = pred-target_m.T
        g_coeffs = (diff.T * data_m.T).item(0)/m
        g_intercept = np.sum(diff)/m
    return g_coeffs, g_intercept
        
def linreg_gd(data, target, params, cost_func, grad_func):
    """
    Perform iters gradient descent to fit data and minimize cost_func
    Args:
    data (ndarray(m,n) or (m,)) : feature(s) or variable(s)
    target (ndarray(m,))        : target variable
    cost_func (function)        : cost function J(w,b) of data to be minimize
    grad_func (function)        : function to compute gradient for the given w,b
    !!params (dict) should contains these following keys!!
    w_in (ndarray(n,) or scalar): initial coefficient(s) - model parameter(s)
    b_in (scalar)               : initial intercept - model parameter
    alpha (scalar)              : learning rate of gradient descent
    iters (int)                 : number of iterations to perform gradient descent

    Returns:
    w_conv (ndarray(n,) or scalar)  : coefficient(s) after iters gradient descent
    b_conv (scalar)                 : intercept after iters gradient descent
    J_history (ndarray(iters,))     : history of cost function J(w,b) over iterations
    """
    w_conv = params['w_in'].copy()
    b_conv = params['b_in']
    intervals = math.ceil(params['iters']/5)
    if params['iters']<100000:
        J_history = np.zeros((params['iters'],))
    else:
        J_history = np.zeros((100000,))
    for idx in range(params['iters']):
        g_coeffs, g_intercept = grad_func(data, target, w_conv, b_conv)
        if idx<100000:
            J_calc = cost_func(data, target, w_conv, b_conv)
            J_history[idx] = J_calc
        if (idx%intervals)==0:
            print("Iteration {}, Cost: {} \ncoefficients: {}\nintercept: {}\n".format(
                idx, J_calc, w_conv, b_conv))
        w_conv = w_conv - (params['alpha'] * g_coeffs)
        b_conv = b_conv - (params['alpha'] * g_intercept)
    print("Iteration {}, Cost: {} \ncoefficients: {}\nintercept: {}\n".format(
                params['iters'], J_calc, w_conv, b_conv))
    return w_conv, b_conv, J_history

def generate_pred(data, w, b):
    """Generate prediction from data with the given w and b
    Args:
    data (ndarray(m,n) or (m,)) : feature(s) or variable(s)
    target (ndarray(m,))        : target variable
    w (ndarray(n,) or scalar)   : coefficient(s) - model parameter(s)
    b (scalar)                  : intercept - model parameter

    Returns:
    preds (ndarray(m,))         : predicted values from data with the given w,b
    """
    data_m, w_m = np.matrix(data), np.matrix(w)
    try:
        pred = (data_m * w_m.T) + b
    except Exception as e:
        pred = (data_m.T * w_m) + b
    return np.array(pred)
