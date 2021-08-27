#%% 
from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stock_data import StockData
from metrics import mse_by_day
from statsmodels.tsa.arima_model import ARIMA

plt.style.use("seaborn")
data = StockData("train_data.csv")

def evaluate_arima_model(train, symbol_list, order, attr_name='open'):
    train_interval = 21
    val_interval = 5
    agg_inter = 720 # by hour  # 12  by minute
    mses = []
    history = []
    for sym in symbol_list:
        start_idx = 0

        # for normalizing the data
        sym_min = min(getattr(train, attr_name)[sym].to_list())
        sym_max = max(getattr(train, attr_name)[sym].to_list())

        for i in range(3):
            train_data = np.array(train.get_slice(attr_name=attr_name,
                                  day_slice=(start_idx, start_idx+train_interval-1))[sym].to_list())
            train_data = np.mean(train_data.reshape(-1, agg_inter), axis=1)
            train_data = (train_data - sym_min) / sym_max
            history.extend(train_data)
            # print("length of training, ", len(history))
            model = ARIMA(np.array(history), order=order)
            model_fit = model.fit()
            # print(model_fit.summary())
            start_idx += train_interval
            val_data = np.array(train.get_slice(attr_name=attr_name,
                            day_slice=(start_idx, start_idx+val_interval-1))[sym].to_list())
            val_data = np.mean(val_data.reshape(-1, agg_inter), axis=1)
            val_data = (val_data - sym_min) / sym_max
            # print("length of val, ", len(val_data))

            pred = model_fit.forecast(steps=len(val_data))[0]
            mse = mse_by_day(pred, val_data, mod='1hr')
            mses.extend(mse)
            history.extend(val_data)
            start_idx += val_interval
        print(sym, np.mean(mses))
    return mses

#%%  evaluate parameters
def get_best_cfg(model_syms):
    # The (p,d,q) order of the model for the number of AR parameters, 
    # differences, and MA parameters to use.
    # p: The number of lag observations included in the model, also called the lag order.
    # d: The number of times that the raw observations are differenced, also called the degree of differencing.
    # q: The size of the moving average window, also called the order of moving average.
    p_values = [6, 7, 10, 15]
    d_values = [0]
    q_values = [0, 1]
    order_lists = [p_values, d_values, q_values]
    mses_dict = {}
    best_cfg_dict = {}
    import itertools
    for sym in model_syms:
        best_score, best_cfg = float("inf"), None
        for order in itertools.product(*order_lists):
            try:
                mses_dict[order] = evaluate_arima_model(data, sym, order)
                mse = np.mean(mses_dict[order])
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s mse=%.10f' % (order, mse))
            except:
                continue
        print(f'For Symbol {sym} Best ARIMA {best_cfg} MSE={best_score:.7f}')
        best_cfg_dict[sym] = best_cfg
    return best_cfg

#%%
# config_dict = get_best_cfg(model_syms)

config_dict = {'A-F': (6, 0 ,1),
               'A': (6, 0 ,1),
               'B': (6, 0, 0),
               'C': (10, 0, 1),
               'D': (15, 0, 0),
               'E': (15, 0, 0),
               'F': (15, 0, 0),
               'G': (15, 0, 0),
               'H': (10, 0, 0),
               'I': (6, 0, 1),
               'J': (6, 0, 1)}
#%%
attr_name = 'open'
agg_inter = 720
# model_syms = [['A', 'B', 'C', 'D', 'E', 'F'], ['G'], ['H'], ['I'], ['J']]
model_syms = [['A'], ['B'], ['C'], ['D'], ['E'], ['F'], ['G'], ['H'], ['I'], ['J']]

for symbols in model_syms:
    # for test
    history = []
    for sym in symbols:
        sym_min = min(getattr(data, attr_name)[sym].to_list())
        sym_max = max(getattr(data, attr_name)[sym].to_list())
        train_data = np.array(data.get_slice(attr_name=attr_name,
                            day_slice=(0, 77))[sym].to_list())
        train_data = np.mean(train_data.reshape(-1, agg_inter), axis=1)
        train_data = (train_data - sym_min) / sym_max
        history.extend(train_data)
    if len(symbols) > 1:
        sym = 'A-F'
    model = ARIMA(np.array(history), config_dict[sym])
    model_fit = model.fit()
    for sym in symbols:
        sym_min = min(getattr(data, attr_name)[sym].to_list())
        sym_max = max(getattr(data, attr_name)[sym].to_list())
        pred = model_fit.forecast(steps=63)[0]
        pred = pred * sym_max + sym_min
        np.save(f'arima_ind_{sym}_pred.npy', pred)

    # for real
    history = []
    for sym in symbols:
        sym_min = min(getattr(data, attr_name)[sym].to_list())
        sym_max = max(getattr(data, attr_name)[sym].to_list())
        train_data = np.array(data.get_slice(attr_name=attr_name,
                            day_slice=(0, 86))[sym].to_list())
        train_data = np.mean(train_data.reshape(-1, agg_inter), axis=1)
        train_data = (train_data - sym_min) / sym_max
        history.extend(train_data)
    if len(symbols) > 1:
        sym = 'A-F'
    model = ARIMA(np.array(history), config_dict[sym])
    model_fit = model.fit()
    for sym in symbols:
        sym_min = min(getattr(data, attr_name)[sym].to_list())
        sym_max = max(getattr(data, attr_name)[sym].to_list())
        pred = model_fit.forecast(steps=63)[0]
        pred = pred * sym_max + sym_min
        np.save(f'arima_ind_{sym}_pred_real.npy', pred)

# %%
