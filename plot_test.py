
#%%
import numpy as np
import matplotlib.pyplot as plt
from stock_data import StockData
from metrics import mse_by_day

# plot test before submission
plt.style.use("seaborn")
data = StockData("train_data.csv")

#%% Comparision of performance before submission
from merge_data import expand_pred
from metrics import mse_by_day
import seaborn as sns
methods = ['baseline', 'arima', 'arima_ind', 'rf', 'lstm']
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
attr_name = 'open'
fig, axs = plt.subplots(2, 5, figsize=(70, 30))
idx = 0
for sym in data.symbol_list:
    real = np.array(data.get_slice(attr_name=attr_name,
                    day_slice=(78, 86))[sym].to_list())
    row = idx // 5
    col = idx % 5
    axs[row, col].plot(real, label="real stock price")
    for method in methods:
        pred = np.load(f'{method}_{sym}_pred.npy')
        pred = expand_pred(pred, 720)
        mses = mse_by_day(real=real, pred=pred, mod='5sec')
        axs[row, col].plot(pred, label=f'{method} pred price p1_mse={np.mean(mses[0:4]): .3f} p2_mse={np.mean(mses[4:]): .3f}')
    axs[row, col].set_title(f'{sym} method comparison', fontsize=36)
    axs[row, col].legend()
    idx += 1
for ax in axs.flat:
    ax.set(xlabel='hours', ylabel='price')
# fig.suptitle('Comparison of different methods on training test data', fontsize=20)
plt.tight_layout()
plt.savefig(f'test_fake.png')
plt.close('all')

#%% Comparision of performance after submission
from evaluation_utils import Evaluator
eval = Evaluator('test_solutions.csv')
from merge_data import expand_pred
from metrics import mse_by_day
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
attr_name = 'open'
fig, axs = plt.subplots(2, 5, figsize=(70, 30))
idx = 0
for sym in data.symbol_list:
    ind = eval.solutions[eval.solutions.index.str.contains(sym)]
    real = np.array(ind[ind.day == 0].open.to_list())
    for i in range(1, 10):
        real = np.append(real, np.array(ind[ind.day == i].open.to_list()))
    print(real.shape)
    row = idx // 5
    col = idx % 5
    axs[row, col].plot(real, label="real stock price")
    for method in methods:
        pred = np.load(f'{method}_{sym}_pred_real.npy')
        pred = expand_pred(pred, 720)
        print(pred.shape)
        mses = mse_by_day(real=real, pred=pred, mod='5sec')
        axs[row, col].plot(pred, label=f'{method} pred price p1_mse={np.mean(mses[0:4]): .3f} p2_mse={np.mean(mses[4:]): .3f}')
    axs[row, col].set_title(f'{sym} method comparison', fontsize=36)
    axs[row, col].legend()
    idx += 1
for ax in axs.flat:
    ax.set(xlabel='hours', ylabel='price')
# fig.suptitle('Comparison of different methods on training test data', fontsize=20)
plt.tight_layout()
plt.savefig(f'test_real.png')
plt.close('all')



#%%
eval.solutions.open.to_list()
# %%
import pandas as pd
submission = pd.read_csv('pred.csv', index_col="id")
submission.rename(columns={'open': 'predicted_open'}, inplace=True)
# %%
submission
# %%
joint_data = pd.concat([eval.solutions, submission], axis=1)

# %%
joint_data.open.to_list()
# %%
joint_data.predicted_open

#%%

no_j = joint_data[joint_data.index.str.contains("A")]

#%%
import matplotlib.pyplot as plt
length = len(no_j.open.to_list())
plt.plot(no_j.open.to_list(), 'b')
plt.plot(no_j.predicted_open.to_list(), 'r')
plt.show()

#%%
import numpy as np
no_j.loc[no_j.predicted_open.isnull(), "predicted_open"] = 0.
no_j["error"] = (no_j.predicted_open - no_j.open)**2
# n.b. I didn't divide by 10 in the eval equation in the instructions.
daily_avg_error = 10*no_j.groupby(["period", "day"]).error.apply(
    np.nanmean
)
period_errors = daily_avg_error.groupby("period").mean()

#%%
period_errors
# return period_errors.to_dict()
# %%
import matplotlib.pyplot as plt
length = len(joint_data.open.to_list())
plt.plot(joint_data.open.to_list()[: length - 45360], 'b')
plt.plot(joint_data.predicted_open.to_list()[: length - 45360], 'r')
plt.show()
# %%
no_j_open = joint_data.open.to_list()[: length - 45360]
no_j_pred = joint_data.predicted_open.to_list()[: length - 45360]