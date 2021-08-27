import numpy as np
import matplotlib.pyplot as plt
from stock_data import StockData
from metrics import mse_by_day
import json
# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1):
  train_x = []
  train_y = []
  for i in range(n_in, len(data)-n_out):
    train_x.append(data[i-n_in: i])
    train_y.append(data[i: i+n_out])
  return np.array(train_x), np.array(train_y)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def select_best_params(train_x, train_y, attr_name, sym, n_in, n_out):
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
  rf = RandomForestRegressor()
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  model_fit = rf_random.fit(train_x, train_y)
  return rf_random.best_params_

def train_rf(best_param, train_x, train_y):
  rf_model = RandomForestRegressor(**best_param)
  rf_model.fit(train_x, train_y)
  return rf_model

def inference_rf(rf_model, in_n, pred_len, train_y):
  pred_y = train_y.copy()
  pred = []
  for i in range(pred_len):
    test_input = pred_y[-in_n:].reshape(1, -1)
    cur = rf_model.predict(test_input)
    pred.append(cur[0])
    np.append(pred_y, cur)
  return pred

from evaluation_utils import Evaluator
eval = Evaluator('test_solutions.csv')
def rf_loop(train, sym, attr_name, n_in, n_out):
    # for test before submission
    train_data = np.array(train.get_slice(attr_name=attr_name,
                    day_slice=(0, 77))[sym].to_list())
    train_data = np.mean(train_data.reshape(-1, agg_inter), axis=1)
    print("length of training, ", len(train_data))
    train_x, train_y = series_to_supervised(train_data, n_in=n_in, n_out=n_out)
    # select hyper-parameters
    best_params = select_best_params(train_x, train_y, attr_name, sym, n_in=n_in, n_out=n_out)
    # save the best hyper parameters 
    with open(f'rf_params_{sym}.txt', 'w') as outfile:
        json.dump(best_params, outfile)
    # train the model
    model = train_rf(best_params, train_x, train_y)
    # do the inference
    pred = inference_rf(model, n_in, 63, train_y)
    np.save(f"rf_{sym}_pred_test.npy", pred)

    test_data = np.array(train.get_slice(attr_name=attr_name,
                        day_slice=(78, 86))[sym].to_list())
    test_data = np.mean(test_data.reshape(-1, agg_inter), axis=1)
    mses = mse_by_day(test_data, np.array(pred), mod='1hr')
    plt.plot(test_data, color = 'black', label = 'stock price')
    plt.plot(pred, color='red', label='stock prediction')
    plt.xlabel('time')
    plt.ylabel('stock price')
    plt.title(f'Random Forest {sym} stock prediction mse={np.mean(mses):.2f}')
    plt.legend()
    plt.savefig(f'rf_{sym}_pred.png')
    plt.close('all')

    # for real submission
    train_data = np.array(train.get_slice(attr_name=attr_name,
                    day_slice=(0, 86))[sym].to_list())
    train_data = np.mean(train_data.reshape(-1, agg_inter), axis=1)
    print("length of training for real, ", len(train_data))

    train_x, train_y = series_to_supervised(train_data, n_in=n_in, n_out=n_out)
    model = train_rf(best_params, train_x, train_y)
    pred = inference_rf(model, n_in, 63, train_y)
    np.save(f"rf_{sym}_pred_real.npy", pred)

    ind = eval.solutions[eval.solutions.index.str.contains(sym)]
    real = np.array(ind[ind.day == 0].open.to_list())
    for i in range(1, 10):
        real = np.append(real, np.array(ind[ind.day == i].open.to_list()))
    test_data  = np.mean(real.reshape(-1, 720), axis=1)
    mses = mse_by_day(test_data, np.array(pred), mod="1hr")

    plt.plot(test_data, color = 'black', label = 'stock price')
    plt.plot(pred, color='red', label='stock prediction')
    plt.title(f'Random Forest {sym} stock prediction mse={np.mean(mses):.2f}')
    plt.xlabel('time')
    plt.ylabel('stock price')
    plt.legend()
    plt.savefig(f'./rf_{sym}_pred_real.png')
    plt.close('all')
    return 
#%%
data = StockData('train_data.csv')
agg_inter = 720 # by hour  # 12  by minute

attr_name = 'open'
n_in = 10
n_out = 1
for sym in data.symbol_list:
    rf_loop(data, sym, attr_name, n_in, n_out)
