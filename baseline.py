from stock_data import StockData
import numpy as np
data = StockData('train_data.csv')

attr_name = 'open'
for sym in data.symbol_list:
    train_data = np.array(data.get_slice(attr_name=attr_name, day_slice=(0, 77))[sym].to_list())
    train_base = np.full(63, train_data[-1])
    np.save(f'baseline_{sym}_pred.npy', train_base)
    train_data = data.open[sym].to_list()[-1]
    real_base = np.full(63, train_data)
    np.save(f'baseline_{sym}_pred_real.npy', real_base)

