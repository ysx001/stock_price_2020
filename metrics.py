import numpy as np
def mse_by_day(real, pred, mod):
    if mod == '5sec':
        day = 5040
    elif mod == '1min':
        day = 8 * 60
    elif mod == '1hr':
        day = 7
    real = real.copy().reshape(-1, day)
    pred = pred.copy().reshape(-1, day)
    mse_by_day = np.nanmean((pred - real)**2, axis=1)
    # print("mse_by_day", mse_by_day)
    # mse = np.mean(mse_by_day)
    # print("mean mse", mse)
    return mse_by_day