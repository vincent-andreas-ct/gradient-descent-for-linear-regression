# import dependency
import numpy as np

def max_scaling(data):
    """
    Performs max scaling on data (only apply for positive values)
    Args:
    data (ndarray(m,n) or (m,))       : data to be scaled with n-features and/or
                                        m-rows

    Returns:
    scaled_data (ndarray(m,n) or (m,)): scaled data using max scaling with
                                        n-features and/or m-rows
    max_$ (ndarray(n,) or scalar)     : maximum value(s) in data (column-wise, 
                                        if data is in 2-D array)
    """
    try:
        m, n = data.shape
        max_column = np.array([max(data[:, idx]) for idx in range(n)])
        scaled_data = np.zeros((m, n))
        for idx in range(m):
            scaled_data[idx, :] = scaled_data[idx, :] + (data[idx, :]/max_column)
        return (max_column, scaled_data)
    except Exception as e:
        max_data = max(data)
        scaled_data = data/max_data
        return (max_data, scaled_data)

def min_max_scaling(data):
    """
    Performs min max scaling on data
    Args:
    data (ndarray(m,n) or (m,))       : data to be scaled with n-features and/or
                                        m-rows

    Returns:
    scaled_data (ndarray(m,n) or (m,)): scaled data using min max scaling with
                                        n-features and/or m-rows
    min_$ (ndarray(n,) or scalar)     : minimum value(s) in data (column-wise,
                                        if data is in 2-D array)
    range_$ (ndarray(n,) or scalar)   : range(s) of values in data (column-wise, 
                                        if data is in 2-D array)
    """
    try:
        m, n = data.shape
        max_column = np.array([max(data[:, idx]) for idx in range(n)])
        min_column = np.array([min(data[:, idx]) for idx in range(n)])
        range_column = max_column - min_column
        scaled_data = np.zeros((m, n))
        for idx in range(m):
            scaled_data[idx, :] += (data[idx, :]-min_column)/range_column
        return (min_column, range_column, scaled_data)
    except Exception as e:
        max_data = max(data)
        min_data = min(data)
        range_data = max_data - min_data
        scaled_data = (data-min_data)/range_data
        return (min_data, range_data, scaled_data)

def zscore_norm(data):
    """
    Performs z score normalization on data
    Args:
    data (ndarray(m,n) or (m,))       : data to be scaled with n-features and/or
                                        m-rows
    Returns:
    scaled_data (ndarray(m,n) or (m,)): scaled data using z score normalization
                                        with n-features and/or m-rows
    mean_$ (ndarray(n,) or scalar)    : mean(s) of data (column-wise, if data is
                                        in 2-D array)
    std_$ (ndarray(n,) or scalar)     : standard deviation(s) of data (column-
                                        wise, if data is in 2-D array)
    """
    try:
        m, n = data.shape
        mean_column = np.array([np.mean(data[:, idx]) for idx in range(n)])
        std_column = np.array([np.std(data[:, idx]) for idx in range(n)])
        scaled_data = np.zeros((m, n))
        for row in range(m):
            scaled_data[row, :] += (data[row, :]-mean_column)/std_column
        return (mean_column, std_column, scaled_data)
    except Exception as e:
        mean_data = np.mean(data)
        std_data = np.std(data)
        scaled_data = (data-mean_data)/std_data
        return (mean_data, std_data, scaled_data)

def mean_norm(data):
    """
    Performs mean normalization on data
    Args:
    data (ndarray(m,n) or (m,))       : data to be scaled with n-features and/or
                                        m-rows
    Returns:
    scaled_data (ndarray(m,n) or (m,)): scaled data using z score normalization
                                        with n-features and/or m-rows
    mean_$ (ndarray(n,) or scalar)    : mean(s) of data (column-wise, if data is
                                        in 2-D array)
    range_$ (ndarray(n,) or scalar)   : range(s) of values in data (column-wise,
                                        if data is in 2-D array)
    """
    try:
        m, n = data.shape
        mean_column = np.array([np.mean(data[:, idx]) for idx in range(n)])
        range_column = np.array([
            max(data[:, idx])-min(data[:, idx])
            for idx in range(n)])
        scaled_data = np.zeros((m, n))
        for row in range(m):
            scaled_data[row, :] += (data[row, :]-mean_column)/range_column
        return (mean_column, range_column, scaled_data)
    except Exception as e:
        mean_data = np.mean(data)
        range_data = max(data)-min(data)
        scaled_data = (data-mean_data)/range_data
        return (mean_data, range_data, scaled_data)
