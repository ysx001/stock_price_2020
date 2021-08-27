import numpy as np
import pandas as pd
def interp_length(p1, p2, length):
    dp = (p2 - p1) / length
    res = np.zeros(length)
    for i in range(length):
        res[i] = p1 + i * dp
    return res

def expand_pred(pred, aggr_int):
    res = np.zeros(len(pred) * aggr_int)
    for i in range(1, len(pred) + 1):
        if i == len(pred):
            res[(i-1)*aggr_int: i*aggr_int] = \
                interp_length(pred[i-1], pred[i-1], aggr_int)
        else:
            res[(i-1)*aggr_int: i*aggr_int] = \
                interp_length(pred[i-1], pred[i], aggr_int)
    return res

# aggr_int = 720
# days = 9
# input = pd.read_csv("train_data.csv")
# symbol_list = np.sort(input.symbol.unique())
# time_list = input.time.unique()
# print("data loaded")
# pred_dict = {}
# for sym in symbol_list:
#     print(sym)
#     pred_data = np.load(f"submit_{sym}_pred_10.npy")
#     pred_data = expand_pred(pred_data, aggr_int)
#     for i in range(days):
#         for t in range(len(pred_data)):
#             t = t % len(time_list)
#             name = f"{sym}-{i}-{time_list[t]}"
#             pred_dict[name] = pred_data[t]
#         print(f"finished day={i}")

# df = pd.DataFrame.from_dict(pred_dict, orient='index', columns=['open'])
# df.index.name = 'id'
# df.to_csv("pred_10.csv")