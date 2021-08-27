#%%
import numpy as np
import matplotlib.pyplot as plt
from stock_data import StockData
from metrics import mse_by_day
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def get_lstm_model(train_len, pred_len):
    model = Sequential()
    model.add(LSTM(units=pred_len,
                   return_sequences=True,
                   input_shape=(train_len, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=pred_len, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=pred_len, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=pred_len))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#%%
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

#%%
plt.style.use("seaborn")
data = StockData("train_data.csv")

#%%
# length = 21
train_len = 63
pred_len = 63
attr_name = 'open'
epoch = 100
model = get_lstm_model(train_len, pred_len)
model.summary()

#%%
all_x = None
all_y = None
for sym in ['A', 'B', 'C', 'D', 'E', 'F']:
    train_data = get_aggr_data(data, attr_name, (0, 77), sym, 720)
    X_train, Y_train = format_training_data(train_data,
                                            train_len=train_len,
                                            pred_len=pred_len)
    
    all_x = X_train if all_x is None else np.concatenate(all_x, X_train)
    all_y = Y_train if all_x is None else np.concatenate(all_x, Y_train)

print(all_x.shape, all_y.shape)

#%%
model.fit(X_train, Y_train, epochs=epoch, batch_size=32)


for sym in ['A', 'B', 'C', 'D', 'E', 'F']:
    test_data = get_aggr_data(data, attr_name, (78, 86), sym, 720)
    X_test = format_test_data(train_data, train_len=train_len)

    inputs = train_data[len(train_data) - train_len:]
    inputs = inputs.reshape(1, -1, 1)
    pred = model.predict(inputs).flatten()


    sym_min = min(getattr(data, attr_name)[sym].to_list())
    sym_max = max(getattr(data, attr_name)[sym].to_list())

    test_data = test_data * sym_max + sym_min
    pred = pred * sym_max + sym_min

    mses = mse_by_day(test_data, pred, mod="1hr")

    plt.plot(test_data, color = 'black', label = 'stock price')
    plt.plot(pred, color='red', label='stock price prediction')
    plt.title(f'{sym} stock price prediction mse={np.mean(mses):.2f}')
    plt.xlabel('time')
    plt.ylabel('stock price')
    plt.legend()
    plt.savefig(f'./lstm_{sym}_pred_af.png')
    plt.close('all')
