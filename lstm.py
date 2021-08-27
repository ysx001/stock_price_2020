#%%
import numpy as np
import matplotlib.pyplot as plt
from stock_data import StockData
from metrics import mse_by_day
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def get_lstm_model(train_len, pred_len, num_features=1):
    model = Sequential()
    model.add(LSTM(units=pred_len,
                   return_sequences=True,
                   input_shape=(train_len, num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=pred_len, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=pred_len, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=pred_len))
    model.add(Dropout(0.2))
    model.add(Dense(units=pred_len))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def format_training_data(data, train_len, pred_len):
    X_train = []
    Y_train = []
    for i in range(train_len, len(data)-pred_len):
        X_train.append(data[i-train_len: i])
        Y_train.append(data[i: i+pred_len])
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array(Y_train)
    return X_train, Y_train

def format_test_data(train_data, train_len):
    inputs = train_data[len(train_data) - train_len:]
    X_test = np.array(inputs).reshape(1, -1, 1)
    return X_test

def get_aggr_data(data, attr_name, day_slice, sym, agg_inter):
    sym_min = min(getattr(data, attr_name)[sym].to_list())
    sym_max = max(getattr(data, attr_name)[sym].to_list())
    train_data = np.array(data.get_slice(attr_name=attr_name,
                                day_slice=day_slice)[sym].to_list())
    train_data  = np.mean(train_data.reshape(-1, agg_inter), axis=1)
    train_data = (train_data - sym_min) / sym_max
    return train_data

def get_all_training_data(data, attr_name, day_slice, train_len, pred_len):
    train_data = get_aggr_data(data, attr_name, day_slice, sym, 720)
    X_train, Y_train = format_training_data(train_data,
                                            train_len=train_len,
                                            pred_len=pred_len)
    
    train_data_first_deri = get_aggr_data(data, attr_name, day_slice, f'{sym}_first_deri', 720)
    X_train_first_deri, _ = format_training_data(train_data_first_deri,
                                                 train_len=train_len,
                                                 pred_len=pred_len)

    train_data_second_deri = get_aggr_data(data, attr_name, day_slice, f'{sym}_second_deri', 720)
    X_train_second_deri, _ = format_training_data(train_data_second_deri,
                                                  train_len=train_len,
                                                  pred_len=pred_len)
    X_train_all = np.stack([X_train, X_train_first_deri, X_train_second_deri], axis=2)
    X_train_all = np.squeeze(X_train_all, axis=3)

    inputs = train_data[len(train_data) - train_len:]
    inputs = inputs.reshape(1, -1, 1)
    inputs_first_deri = train_data_first_deri[len(train_data_first_deri) - train_len:]
    inputs_first_deri = inputs_first_deri.reshape(1, -1, 1)
    inputs_second_deri = train_data_second_deri[len(train_data_second_deri) - train_len:]
    inputs_second_deri = inputs_second_deri.reshape(1, -1, 1)
    all_inputs = np.stack([inputs, inputs_first_deri, inputs_second_deri], axis=2)
    all_inputs = np.squeeze(all_inputs, axis=3)

    return X_train_all, Y_train, all_inputs



#%%
plt.style.use("seaborn")
data = StockData("train_data.csv")

# #%%
# sym = 'J'
#%%


# from merge_data import expand_pred
# pred_data = np.load(f"submit_{sym}_pred_new.npy")
# aggr_int = 720
# pred_data = expand_pred(pred_data, aggr_int)


# all_base = np.append(data.open[sym].to_list(), np.full(45360, data.open[sym][-1]))
# all_real = np.append(data.open[sym].to_list(), real)
# all_pred = np.append(data.open[sym].to_list(), pred_data)

# plt.plot(all_base, 'r')
# plt.plot(all_real, 'b')
# plt.plot(all_pred, 'g')

#%%
# length = 21
from evaluation_utils import Evaluator
eval = Evaluator('test_solutions.csv')

train_len = 63
pred_len = 63
attr_name = 'open'
epoch = 100
for sym in data.symbol_list:
    # before submission
    ind = eval.solutions[eval.solutions.index.str.contains(sym)]
    X_train_all, Y_train, all_inputs = get_all_training_data(data, attr_name, (0, 77), train_len, pred_len)
    print(X_train_all.shape, Y_train.shape, all_inputs.shape)

    model = get_lstm_model(train_len, pred_len, num_features=3)
    model.summary()
    model.fit(X_train_all, Y_train, epochs=epoch, batch_size=32)

    sym_min = min(getattr(data, attr_name)[sym].to_list())
    sym_max = max(getattr(data, attr_name)[sym].to_list())
    mean = np.mean(getattr(data, attr_name)[sym].to_list())
    
    pred = model.predict(all_inputs).flatten()
    pred = pred * sym_max + sym_min
    print(pred.shape)
    np.save(f"lstm_3_feat_{sym}_pred.npy", pred)
    
    X_train_all, Y_train, all_inputs = get_all_training_data(data, attr_name, (0, 86), train_len, pred_len)

    model = get_lstm_model(train_len, pred_len, num_features=3)
    # model.summary()
    model.fit(X_train_all, Y_train, epochs=epoch, batch_size=32)
    
    pred = model.predict(all_inputs).flatten()
    pred = pred * sym_max + sym_min
    
    np.save(f"lstm_3_feat_{sym}_pred_real.npy", pred)


    # real = np.array(ind[ind.day == 0].open.to_list())
    # for i in range(1, 10):
    #     real = np.append(real, np.array(ind[ind.day == i].open.to_list()))
    # test_data  = np.mean(real.reshape(-1, 720), axis=1)
    # mses = mse_by_day(test_data, pred, mod="1hr")

    # plt.plot(test_data, color = 'black', label = 'stock price')
    # plt.plot(pred, color='red', label='stock price prediction')
    # plt.title(f'{sym} stock price prediction mse={np.mean(mses):.2f}')
    # plt.xlabel('time')
    # plt.ylabel('stock price')
    # plt.legend()
    # plt.savefig(f'./lstm_{sym}_pred_real_epoch_{epoch}.png')
    # plt.close('all')

# %%
