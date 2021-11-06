import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.lines as mlines
import random

# get_ipython().run_line_magic('matplotlib', 'inline')
cmap = cm.tab10

# %%
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# %%
# for did in ['ZHWX33', 'ZHWX34']:
print('start')
data = pd.read_csv('data/ZHWX33.csv')

data = data.append(pd.read_csv('data/ZHWX34.csv'))
data['time'] = pd.to_datetime(data['time'])
stime = pd.to_datetime('2019-10-01 00:00:00')
data['day'] = np.int64((data['time'] - stime).dt.total_seconds() // (24 * 3600) + 1)
data['min'] = data['time'].dt.hour * 60 + data['time'].dt.minute
data = data[(data['day'] >= 1) & (data['day'] <= 21)]
data = data.sort_values(by=['did', 'laneno', 'day', 'min']).reset_index(drop=True)
# data.head()
data.loc[(data['occ'] < 5) & (data['spd'] == 0), 'spd'] = data['spd'].quantile(0.9)


def impute(df):
    data1 = df.copy().reset_index(drop=True)
    datasample = data1.copy()
    tsize = (len(set(datasample['laneno'])), 21, 1440)
    # %%
    np.random.seed(0)

    random_mat = np.random.random(size=np.int(datasample.shape[0] / T))
    binary_mat = np.round(random_mat + 0.5 - missing_rate)
    binary_mat = np.repeat(binary_mat, T)
    r = np.where((data1[p].values != 0) & (binary_mat == 0))[0].tolist()  # 缺失数据位置
    # r = np.where((binary_mat==0))[0].tolist()
    # r = random.sample(set(datasample.index), int(rate*datasample.shape[0]))
    # r = []
    # r[:5]

    # %%
    # data1['bin'] = binary_mat

    # %%
    # len(r)/datasample.shape[0]

    # %%
    origin = datasample.loc[r].copy()
    # origin.to_csv('data/%s_%s_%.2f.csv'%(p, itype, rate), index=False)
    # origin.head()

    # %%
    datasample.loc[r, p] = np.nan
    datasample.loc[r, p].head()

    # %%
    lanes = list(set(datasample['laneno']))
    # lanes

    # %%
    # 末次观测值修复
    from impyute import locf
    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[['day', 'min', 'laneno', p]][datasample['laneno'] == lane].copy().sort_values(
            by=['day', 'min'])
        res = locf(temp[p].values.reshape(-1, 1), axis=1)
        temp[p] = np.round(res, 1)
        if i == 0:
            allres = temp
        else:
            allres = allres.append(temp)

    temp = datasample[['day', 'min', 'laneno', ]].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['simple'] = temp[p].values
    # origin.head()

    # %%
    # mean
    temp = datasample[['laneno', 'min']].loc[r]
    mean = pd.merge(temp.drop_duplicates(), datasample[['laneno', 'min', p]], on=['laneno', 'min'], how='inner')
    mean = mean.groupby(['laneno', 'min']).apply(lambda df: pd.Series({'mean': df[p].mean(), 'median': df[p].median()}))
    temp = pd.merge(temp, mean, on=['laneno', 'min'], how='left')
    origin['mean'] = np.round(temp['mean'].values, 1)
    origin['median'] = np.round(temp['median'].values, 1)
    # origin.head()

    # %%
    # moving average
    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[['day', 'min', 'laneno', p]][datasample['laneno'] == lane].copy().sort_values(
            by=['day', 'min'])
        res = locf(temp[p].values.reshape(-1, 1), axis=1)
        temp[p] = np.round(res, 1)
        if i == 0:
            allres = temp
        else:
            allres = allres.append(temp)
    allres = pd.merge(datasample[['day', 'min', 'laneno']], allres, on=['day', 'min', 'laneno'], how='left')

    # %%

    for n in [3, 5, 7]:  # 窗口大小
        for i in range(n // 2):
            temp = allres[['laneno', 'day', 'min']].loc[r].reset_index(drop=True)
            temp['min'] = temp['min'] - 1 - i
            a1 = temp[temp['min'] < 0].index
            temp['day'][temp['min'] < 0] = temp['day'][temp['min'] < 0] - 1
            temp['min'] = temp['min'] % 1440
            temp = pd.merge(temp, allres[['laneno', 'day', 'min', p]], on=['laneno', 'day', 'min'], how='left')
            temp.loc[a1, 'min'] = temp.loc[a1, 'min'] - 1440
            temp['min'] = temp['min'] + i + 1
            temp.loc[a1, 'day'] = temp.loc[a1, 'day'] + 1

            if i == 0:
                res = temp[p].values.reshape(-1, 1)
            else:
                res = np.hstack([res, temp[p].values.reshape(-1, 1)])

        for i in range(n // 2):
            temp = allres[['laneno', 'day', 'min']].loc[r].reset_index(drop=True)
            temp['min'] = temp['min'] + 1 + i
            a1 = temp[temp['min'] >= 1440].index
            temp['day'][temp['min'] >= 1440] = temp['day'][temp['min'] >= 1440] + 1
            temp['min'] = temp['min'] % 1440
            temp = pd.merge(temp, allres[['laneno', 'day', 'min', p]], on=['laneno', 'day', 'min'], how='left')
            temp.loc[a1, 'min'] = 1440 - temp.loc[a1, 'min']
            temp['min'] = temp['min'] - i - 1
            temp.loc[a1, 'day'] = temp.loc[a1, 'day'] - 1

            res = np.hstack([res, temp[p].values.reshape(-1, 1)])

        origin['ma' + str(n)] = np.round(np.nanmean(res, axis=1), 1)
    # origin.head()

    # %%
    # exponential
    for alpha in [0.2, 0.5, 0.8]:  # 平滑系数
        temp = datasample[['laneno', 'day', 'min', p]].loc[r]
        res = datasample[['laneno', 'day', 'min', p]].copy()
        for i in range(1, 21):
            temp = pd.merge(temp, res[['laneno', 'day', 'min', p]][res['day'] == i], on=['laneno', 'min'], how='left')
            temp = temp.rename(columns={'day_x': 'day', p + '_x': p})
            #         temp.loc[(pd.isnull(temp[p]))&(temp['day']<=i)&(~pd.isnull(temp[p+'_y'])), p] = temp[p+'_y'][(~pd.isnull(temp[p]))&(temp['day']>i)&(~pd.isnull(temp[p+'_y']))].values
            temp.loc[(pd.isnull(temp[p])), p] = temp[p + '_y'][(pd.isnull(temp[p]))].values
            temp.loc[(~pd.isnull(temp[p])) & (temp['day'] > i) & (~pd.isnull(temp[p + '_y'])), p] = temp[p][(~pd.isnull(
                temp[p])) & (temp['day'] > i) & (~pd.isnull(temp[p + '_y']))].values * (1 - alpha) + temp[p + '_y'][(
                                                                                                                        ~pd.isnull(
                                                                                                                            temp[
                                                                                                                                p])) & (
                                                                                                                                temp[
                                                                                                                                    'day'] > i) & (
                                                                                                                        ~pd.isnull(
                                                                                                                            temp[
                                                                                                                                p + '_y']))].values * alpha

            temp = temp[['laneno', 'day', 'min', p]]

        origin['expo' + str(alpha)] = np.round(temp[p].values, 1)
    # origin.head()

    # %%
    # time regression
    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    # lanes = list(set(datasample['laneno']))

    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[['day', 'min', 'laneno', p]][datasample['laneno'] == lane].copy()
        temp = temp.rename(columns={p: 'origin'}).reset_index(drop=True)

        temp['day'] = temp['day'] - 1
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'lastday'})
        temp['day'] = temp['day'] + 1

        temp['day'] = temp['day'] + 1
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'nextday'})
        temp['day'] = temp['day'] - 1

        temp['min'] = temp['min'] - 2
        a1 = temp[temp['min'] < 0].index
        temp['day'][temp['min'] < 0] = temp['day'][temp['min'] < 0] - 1
        temp['min'] = temp['min'] % 1440
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'last2min'})
        temp.loc[a1, 'min'] = temp.loc[a1, 'min'] - 1440
        temp['min'] = temp['min'] + 2
        temp.loc[a1, 'day'] = temp.loc[a1, 'day'] + 1

        temp['min'] = temp['min'] - 1
        a1 = temp[temp['min'] < 0].index
        temp['day'][temp['min'] < 0] = temp['day'][temp['min'] < 0] - 1
        temp['min'] = temp['min'] % 1440
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'lastmin'})
        temp.loc[a1, 'min'] = temp.loc[a1, 'min'] - 1440
        temp['min'] = temp['min'] + 1
        temp.loc[a1, 'day'] = temp.loc[a1, 'day'] + 1

        temp['min'] = temp['min'] + 1
        a2 = temp[temp['min'] >= 1440].index
        temp['day'][temp['min'] >= 1440] = temp['day'][temp['min'] >= 1440] + 1
        temp['min'] = temp['min'] % 1440
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'nextmin'})
        temp.loc[set(a2), 'min'] = 1440 - temp.loc[set(a2), 'min']
        temp['min'] = temp['min'] - 1
        temp.loc[set(a2), 'day'] = temp.loc[set(a2), 'day'] - 1

        X = temp.iloc[:, 3:].values
        estimator = LinearRegression()
        imputer = IterativeImputer(estimator=estimator)
        imputer.fit(X)
        temp.iloc[:, 3:] = np.round(imputer.fit_transform(X), 1)
        temp = temp.rename(columns={'origin': p})
        res = temp.iloc[:, :4].copy()

        if i == 0:
            allres = res
        else:
            allres = allres.append(res)

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['t'] = temp[p].values
    # origin.head()
    # %%
    # 相邻车道
    temp = datasample[['day', 'min', 'laneno']].loc[r]
    for i in range(len(lanes)):
        lane = lanes[i]
        if i == 0:
            temp = datasample[['day', 'min', p]][datasample['laneno'] == lane].copy()
        else:
            temp = pd.merge(temp, datasample[['day', 'min', p]][datasample['laneno'] == lane], on=['day', 'min'],
                            how='left')
        temp = temp.rename(columns={p: p + '_' + str(lane)})
    temp[p] = np.round(np.nanmean(temp.iloc[:, 2:], axis=1), 1)
    res = temp[['day', 'min', p]].copy()
    temp = datasample[['day', 'min']].loc[r]
    temp = pd.merge(temp, res, on=['day', 'min'], how='left')
    origin['lane'] = temp[p].values
    # origin.head()

    # %%
    # 断面车道
    # explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer

    # %%
    # correlation  both
    from sklearn.linear_model import LinearRegression

    # lanes = list(set(datasample['laneno']))
    for i in range(len(lanes)):
        lane = lanes[i]
        if i == 0:
            temp = datasample[['day', 'min', 'laneno', p]][datasample['laneno'] == lane].copy()
        else:
            temp = pd.merge(temp, datasample[['day', 'min', p]][datasample['laneno'] == lane], on=['day', 'min'],
                            how='left')
        temp = temp.rename(columns={p: p + '_' + str(lane)})

    X = temp.iloc[:, 3:].values

    estimator = LinearRegression()
    imputer = IterativeImputer(estimator=estimator)
    imputer.fit(X)
    temp.iloc[:, 3:] = imputer.fit_transform(X)
    res = temp.iloc[:, :4].copy()
    for i in range(1, len(lanes)):
        temp['laneno'] = lanes[i]
        temp.iloc[:, 3] = temp.iloc[:, 3 + i]
        res = res.append(temp.iloc[:, :4])

    res = res.rename(columns={p + '_' + str(lanes[0]): p})

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, res, on=['day', 'min', 'laneno'], how='left')
    origin['alllanes'] = np.round(temp[p].values, 1)
    # origin.head()

    # %%
    # np.sum(pd.isnull(origin['alllanes']))/origin.shape[0]

    # %%
    # correlation  multi 时空多元回归
    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    # lanes = list(set(datasample['laneno']))

    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[['day', 'min', 'laneno', p]][datasample['laneno'] == lane].copy()
        temp = temp.rename(columns={p: 'origin'})
        temp['laneno'] = temp['laneno'] - 1
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'left'})
        temp['laneno'] = temp['laneno'] + 2
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'right'})
        temp['laneno'] = temp['laneno'] - 1
        temp = temp.dropna(axis=1, how='all')

        temp['day'] = temp['day'] - 1
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'lastday'})
        temp['day'] = temp['day'] + 1

        temp['min'] = temp['min'] - 1
        a1 = temp[temp['min'] < 0].index
        temp['day'][temp['min'] < 0] = temp['day'][temp['min'] < 0] - 1
        temp['min'] = temp['min'] % 1440
        temp = pd.merge(temp, datasample[['day', 'min', 'laneno', p]], on=['day', 'min', 'laneno'], how='left')
        temp = temp.rename(columns={p: 'lastmin'})
        temp.loc[a1, 'min'] = -1
        temp['min'] = temp['min'] + 1
        temp.loc[a1, 'day'] = temp.loc[a1, 'day'] + 1

        X = temp.iloc[:, 3:].values
        estimator = LinearRegression()
        imputer = IterativeImputer(estimator=estimator)
        imputer.fit(X)
        temp.iloc[:, 3:] = np.round(imputer.fit_transform(X), 1)
        temp = temp.rename(columns={'origin': p})
        res = temp.iloc[:, :4].copy()

        if i == 0:
            allres = res
        else:
            allres = allres.append(res)

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['multi'] = temp[p].values
    # origin.head()

    # %% [markdown]
    # ### BTMF

    # %%
    # 矩阵修复 https://github.com/xinychen/transdim
    # 我觉得效果不是很好 比较新的方法可以试一下
    # import numpy as np
    from numpy.random import multivariate_normal as mvnrnd
    from scipy.stats import wishart
    from scipy.stats import invwishart
    from numpy.linalg import inv as inv

    # %%
    def kr_prod(a, b):
        return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

    def cov_mat(mat):
        dim1, dim2 = mat.shape
        new_mat = np.zeros((dim2, dim2))
        mat_bar = np.mean(mat, axis=0)
        for i in range(dim1):
            new_mat += np.einsum('i, j -> ij', mat[i, :] - mat_bar, mat[i, :] - mat_bar)
        return new_mat

    def ten2mat(tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

    def mat2ten(mat, tensor_size, mode):
        index = list()
        index.append(mode)
        for i in range(tensor_size.shape[0]):
            if i != mode:
                index.append(i)
        return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order='F'), 0, mode)

    def mnrnd(M, U, V):
        """
        Generate matrix normal distributed random matrix.
        M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
        """
        dim1, dim2 = M.shape
        X0 = np.random.rand(dim1, dim2)
        P = np.linalg.cholesky(U)
        Q = np.linalg.cholesky(V)
        return M + np.matmul(np.matmul(P, X0), Q.T)

    # %%
    def BTMF(dense_mat, sparse_mat, init, rank, time_lags, maxiter1, maxiter2):
        """Bayesian Temporal Matrix Factorization, BTMF."""
        W = init["W"]
        X = init["X"]

        d = time_lags.shape[0]
        dim1, dim2 = sparse_mat.shape
        pos = np.where((dense_mat != 0) & (sparse_mat == 0))
        position = np.where(sparse_mat != 0)
        binary_mat = np.zeros((dim1, dim2))
        binary_mat[position] = 1

        beta0 = 1
        nu0 = rank
        mu0 = np.zeros((rank))
        W0 = np.eye(rank)
        tau = 1
        alpha = 1e-6
        beta = 1e-6
        S0 = np.eye(rank)
        Psi0 = np.eye(rank * d)
        M0 = np.zeros((rank * d, rank))

        W_plus = np.zeros((dim1, rank))
        X_plus = np.zeros((dim2, rank))
        X_new_plus = np.zeros((dim2 + 1, rank))
        A_plus = np.zeros((rank, rank, d))
        mat_hat_plus = np.zeros((dim1, dim2 + 1))
        for iters in range(maxiter1):
            print(iters)
            W_bar = np.mean(W, axis=0)
            var_mu_hyper = (dim1 * W_bar) / (dim1 + beta0)
            var_W_hyper = inv(inv(W0) + cov_mat(W) + dim1 * beta0 / (dim1 + beta0) * np.outer(W_bar, W_bar))
            var_Lambda_hyper = wishart(df=dim1 + nu0, scale=var_W_hyper, seed=None).rvs()
            var_mu_hyper = mvnrnd(var_mu_hyper, inv((dim1 + beta0) * var_Lambda_hyper))

            var1 = X.T
            var2 = kr_prod(var1, var1)
            var3 = tau * np.matmul(var2, binary_mat.T).reshape([rank, rank, dim1]) + np.dstack(
                [var_Lambda_hyper] * dim1)
            var4 = (tau * np.matmul(var1, sparse_mat.T)
                    + np.dstack([np.matmul(var_Lambda_hyper, var_mu_hyper)] * dim1)[0, :, :])
            for i in range(dim1):
                inv_var_Lambda = inv(var3[:, :, i])
                W[i, :] = mvnrnd(np.matmul(inv_var_Lambda, var4[:, i]), inv_var_Lambda)
            if iters + 1 > maxiter1 - maxiter2:
                W_plus += W

            Z_mat = X[np.max(time_lags): dim2, :]
            Q_mat = np.zeros((dim2 - np.max(time_lags), rank * d))
            for t in range(np.max(time_lags), dim2):
                Q_mat[t - np.max(time_lags), :] = X[t - time_lags, :].reshape([rank * d])
            var_Psi = inv(inv(Psi0) + np.matmul(Q_mat.T, Q_mat))
            var_M = np.matmul(var_Psi, np.matmul(inv(Psi0), M0) + np.matmul(Q_mat.T, Z_mat))
            var_S = (S0 + np.matmul(Z_mat.T, Z_mat) + np.matmul(np.matmul(M0.T, inv(Psi0)), M0)
                     - np.matmul(np.matmul(var_M.T, inv(var_Psi)), var_M))
            Sigma = invwishart(df=nu0 + dim2 - np.max(time_lags), scale=var_S, seed=None).rvs()
            A = mat2ten(mnrnd(var_M, var_Psi, Sigma).T, np.array([rank, rank, d]), 0)
            if iters + 1 > maxiter1 - maxiter2:
                A_plus += A

            Lambda_x = inv(Sigma)
            var1 = W.T
            var2 = kr_prod(var1, var1)
            var3 = tau * np.matmul(var2, binary_mat).reshape([rank, rank, dim2]) + np.dstack([Lambda_x] * dim2)
            var4 = tau * np.matmul(var1, sparse_mat)
            for t in range(dim2):
                Mt = np.zeros((rank, rank))
                Nt = np.zeros(rank)
                if t < np.max(time_lags):
                    Qt = np.zeros(rank)
                else:
                    Qt = np.matmul(Lambda_x, np.matmul(ten2mat(A, 0), X[t - time_lags, :].reshape([rank * d])))
                if t < dim2 - np.min(time_lags):
                    if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                        index = list(range(0, d))
                    else:
                        index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2)))[0]
                    for k in index:
                        Ak = A[:, :, k]
                        Mt += np.matmul(np.matmul(Ak.T, Lambda_x), Ak)
                        A0 = A.copy()
                        A0[:, :, k] = 0
                        var5 = (X[t + time_lags[k], :]
                                - np.matmul(ten2mat(A0, 0), X[t + time_lags[k] - time_lags, :].reshape([rank * d])))
                        Nt += np.matmul(np.matmul(Ak.T, Lambda_x), var5)
                var_mu = var4[:, t] + Nt + Qt
                if t < np.max(time_lags):
                    inv_var_Lambda = inv(var3[:, :, t] + Mt - Lambda_x + np.eye(rank))
                else:
                    inv_var_Lambda = inv(var3[:, :, t] + Mt)
                X[t, :] = mvnrnd(np.matmul(inv_var_Lambda, var_mu), inv_var_Lambda)
            mat_hat = np.matmul(W, X.T)

            X_new = np.zeros((dim2 + 1, rank))
            if iters + 1 > maxiter1 - maxiter2:
                X_new[0: dim2, :] = X.copy()
                X_new[dim2, :] = np.matmul(ten2mat(A, 0), X_new[dim2 - time_lags, :].reshape([rank * d]))
                X_new_plus += X_new
                mat_hat_plus += np.matmul(W, X_new.T)

            tau = np.random.gamma(alpha + 0.5 * sparse_mat[position].shape[0],
                                  1 / (beta + 0.5 * np.sum((sparse_mat - mat_hat)[position] ** 2)))
            rmse = np.sqrt(np.sum((dense_mat[pos] - mat_hat[pos]) ** 2) / dense_mat[pos].shape[0])
            if (iters + 1) % 200 == 0 and iters < maxiter1 - maxiter2:
                print('Iter: {}'.format(iters + 1))
                print('RMSE: {:.6}'.format(rmse))
                print()

        W = W_plus / maxiter2
        X_new = X_new_plus / maxiter2
        A = A_plus / maxiter2
        mat_hat = mat_hat_plus / maxiter2
        print(mat_hat.shape, dense_mat.shape)
        #     if maxiter1 >= 100:
        final_mape = np.sum(np.abs(dense_mat[pos] - mat_hat[pos]) / dense_mat[pos]) / dense_mat[pos].shape[0]
        final_rmse = np.sqrt(np.sum((dense_mat[pos] - mat_hat[pos]) ** 2) / dense_mat[pos].shape[0])
        print('Imputation MAPE: {:.6}'.format(final_mape))
        print('Imputation RMSE: {:.6}'.format(final_rmse))
        print()

        return mat_hat, W, X_new, A

    # %%
    tensor = data1[p].values.reshape(tsize)
    dense_mat = tensor.reshape([tensor.shape[0], tensor.shape[1] * tensor.shape[2]])
    binary_mat = data1['bin'].values.reshape([tensor.shape[0], tensor.shape[1] * tensor.shape[2]])
    sparse_mat = np.multiply(dense_mat, binary_mat)

    # %%
    import time
    start = time.time()
    dim1, dim2 = sparse_mat.shape
    rank = 10
    time_lags = np.array([1, 2, 3, 1440, 2880])
    init = {"W": 0.1 * np.random.rand(dim1, rank), "X": 0.1 * np.random.rand(dim2, rank)}
    maxiter1 = 10
    maxiter2 = 5
    res, _, _, _ = BTMF(dense_mat, sparse_mat, init, rank, time_lags, maxiter1, maxiter2)
    end = time.time()
    print('Running time: %d seconds' % (end - start))
    temp.shape[0], len(res.reshape(-1, 1)), len(dense_mat.reshape(-1, 1))
    temp = data1[['day', p]].copy()
    temp.loc[:, p] = np.round(res[:, :-1].reshape(-1, 1), 1)
    origin['BTMF'] = temp[p].loc[r]
    origin.head()

    # %% [markdown]
    # #### MICE

    # %%
    # explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer

    # %%
    # allres.head()

    # %%
    estimator = LinearRegression()
    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[datasample['laneno'] == lane].copy()
        imputer = IterativeImputer(estimator=estimator)
        #     temp.loc[temp['occ']>0, 'flow'] = temp.loc[temp['occ']>0, 'flow']/temp.loc[temp['occ']>0, 'occ']
        imputer.fit(temp[['flow', 'spd', 'occ']])
        temp.loc[:, ['flow', 'spd', 'occ']] = imputer.fit_transform(temp[['flow', 'spd', 'occ']])
        #     temp.loc[temp['occ']>0, 'flow'] = temp.loc[temp['occ']>0, 'flow']*temp.loc[temp['occ']>0, 'occ']

        if i == 0:
            allres = temp
        else:
            allres = allres.append(temp)

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['mice'] = np.round(temp[p].values, 1)
    # origin.head()

    # %%

    from sklearn.neighbors import KNeighborsRegressor
    # 归一化
    # estimator = KNeighborsRegressor(n_neighbors=15)
    # imputer = IterativeImputer(estimator=estimator)
    # imputer.fit(scaleddata)
    # res = imputer.fit_transform(scaleddata)
    # outputdata = scaler.inverse_transform(res)
    # origin['KNN_scale'] = np.round(outputdata[r,idx], 1)
    # origin.head()

    # %%
    estimator = KNeighborsRegressor(n_neighbors=5)
    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[datasample['laneno'] == lane].copy()
        imputer = IterativeImputer(estimator=estimator)
        imputer.fit(temp[['flow', 'spd', 'occ']])
        temp.loc[:, ['flow', 'spd', 'occ']] = imputer.fit_transform(temp[['flow', 'spd', 'occ']])

        if i == 0:
            allres = temp
        else:
            allres = allres.append(temp)

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['KNN'] = np.round(temp[p].values, 1)
    # origin.head()

    # %%
    # 决策树
    from sklearn.tree import DecisionTreeRegressor
    # estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0)
    # imputer = IterativeImputer(estimator=estimator)
    # imputer.fit(scaleddata)
    # res = imputer.fit_transform(scaleddata)
    # outputdata = scaler.inverse_transform(res)
    # origin['DTR_scale'] = np.round(outputdata[r,idx], 1)
    # origin.head()

    # %%
    estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0)
    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[datasample['laneno'] == lane].copy()
        imputer = IterativeImputer(estimator=estimator)
        imputer.fit(temp[['flow', 'spd', 'occ']])
        res = imputer.fit_transform(temp[['flow', 'spd', 'occ']])
        temp.loc[:, ['flow', 'spd', 'occ']] = res

        if i == 0:
            allres = temp
        else:
            allres = allres.append(temp)

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['DTR'] = np.round(temp[p].values, 1)
    # origin.head()

    # %%
    # ##scaled
    from sklearn.ensemble import ExtraTreesRegressor
    # extra trees 和randomforest 比较像
    # # estimator = KNeighborsRegressor(n_neighbors=15)
    # imputer = IterativeImputer(estimator=ExtraTreesRegressor())
    # imputer.fit(scaleddata)
    # res = imputer.fit_transform(scaleddata)
    # outputdata = scaler.inverse_transform(res)
    # origin['ET_scaled'] = np.round(outputdata[r,idx], 1)
    # origin.head()

    # %%
    estimator = ExtraTreesRegressor(random_state=0)
    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[['day', 'min', 'laneno', p]][datasample['laneno'] == lane].copy()
        imputer = IterativeImputer(estimator=estimator)
        imputer.fit(temp[['flow', 'spd', 'occ']])
        res = imputer.fit_transform(temp[['flow', 'spd', 'occ']])
        if i == 0:
            alllres = res
        else:
            allres = allres.append(res)

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['ET'] = np.round(temp[p].values, 1)
    origin.head()

    # %%
    ##scaled
    from sklearn.ensemble import RandomForestRegressor
    # 随机森林
    # # estimator = KNeighborsRegressor(n_neighbors=15)
    # imputer = IterativeImputer(estimator=ExtraTreesRegressor())
    # imputer.fit(scaleddata)
    # res = imputer.fit_transform(scaleddata)
    # outputdata = scaler.inverse_transform(res)
    # origin['ET_scaled'] = np.round(outputdata[r,idx], 1)
    # origin.head()

    # %%
    estimator = RandomForestRegressor(random_state=0)
    for i in range(len(lanes)):
        lane = lanes[i]
        temp = datasample[datasample['laneno'] == lane].copy()
        imputer = IterativeImputer(estimator=estimator)
        imputer.fit(temp[['flow', 'spd', 'occ']])
        temp.loc[:, ['flow', 'spd', 'occ']] = imputer.fit_transform(temp[['flow', 'spd', 'occ']])

        if i == 0:
            allres = temp
        else:
            allres = allres.append(temp)

    temp = datasample[['day', 'min', 'laneno']].loc[r]
    temp = pd.merge(temp, allres, on=['day', 'min', 'laneno'], how='left')
    origin['RF'] = np.round(temp[p].values, 1)
    origin.head()
    origin['idx'] = r

    return origin


# %%
for p, idx in [('flow', 0), ('spd', 1), ('occ', 2)]:
    print('params %s' % p)
    # for missing_rate in [0.01, 0.05, 0.1]:
    for missing_rate in [0.1]:  # missing rate 缺失数据比例
        print('rate %.2f' % missing_rate)
        for itype in ['RM-U', 'NM-U']:  # RM-U 随机缺失 NM-U 连续时段缺失 gap 缺失时长
            print('type %s' % itype)
            if itype == 'RM-U':
                gaps = [1]
            else:
                gaps = [20, 60]
            for T in gaps:
                print('period %d' % T)

                origin = data.groupby('did', as_index=False).apply(impute)
                # %%
                # np.sum(pd.isnull(origin), axis=0)/origin.shape[0]

                # %%
                origin.to_csv('data/res1/%s/%s_%d_%.2f.csv' % (itype, p, T, missing_rate), index=False)

                # %%
                if p == 'spd':
                    origin = origin[origin[p] != 0]

                # %%
                # origin1 = origin[origin[p]!=0]

                # %%
                width = len(origin.iloc[0, 10:])
                # width

                # %%
                resmape = np.abs(((origin.iloc[:, 10:].values - np.tile(origin.iloc[:, 4 + idx].values.reshape(-1, 1),
                                                                        (1, width))) / np.tile(
                    origin.iloc[:, 4 + idx].values.reshape(-1, 1), (1, width))))
                mape = np.round(np.nanmean(resmape, axis=0) * 100, 2)
                resrmse = (origin.iloc[:, 10:].values - np.tile(origin.iloc[:, 4 + idx].values.reshape(-1, 1),
                                                                (1, width))) ** 2
                rmse = np.round(np.sqrt(np.nanmean(resrmse, axis=0)), 2)
                methodidx = origin.columns[10:]
                resdf = pd.DataFrame({'mape': mape, 'rmse': rmse}, index=methodidx)
                # resdf

                # %%
                resdf.loc[['simple', 'ma5', 'mean', 'median',
                           'expo0.8', 't', 'lane', 'alllanes', 'multi',
                           'mice', 'KNN', 'DTR', 'RF']].to_csv(
                    'data/eval1/%s/%s_%d_%.2f.csv' % (itype, p, T, missing_rate))

                print('save %s_%s_%d_%.2f.csv' % (itype, p, T, missing_rate))

                # # %%
                # # 画图代码
                # allcols = origin.columns
                # day = 12
                # did = 'ZHWX33'
                # for i in range(10, len(allcols)):
                #     temp = data[data['did']==did].copy().sort_values(by='time').reset_index(drop=True)
                #     temp['x'] = temp[p].values
                #     origin1 = origin[origin['did']==did].sort_values(by='time').reset_index(drop=True)
                #     r = origin1['idx'].values
                #     print(len(r))
                #     temp.loc[r,'x'] = origin1.loc[r,allcols[i]].values
                #     temp = temp[(temp['day']==day)&(temp['laneno']==2)]
                #     tempindex = temp['time'].values
                #     # temp = temp.set_index('time')
                #     rr = set(r)&set(temp.index)

                #     N = 10
                #     aa = temp[p].values.copy()
                #     a = temp[p].values[:-(N-1)].reshape(-1,1)
                #     # i=4
                #     # print(len(temp[p].values[i:-(N-1)+i].reshape(-1,1)))
                #     for i in range(1,N):
                #     #     print(i)
                #         aa[i] = np.mean(temp[p].values[:i+1])
                #         if i == N-1:
                #             a = np.hstack([a,temp[p].values[i:].reshape(-1,1)])
                #         else:
                #             a = np.hstack([a,temp[p].values[i:-(N-1)+i].reshape(-1,1)])
                #     # print(a.shape[0])
                #     a = np.mean(a, axis=1)
                #     aa[N-1:] = a
                #     temp['smooth'] = aa

                #     fig, ax = plt.subplots(1,1,figsize=(8,4))
                #     ax.plot_date(temp['time'],temp['smooth'],fmt='-',lw=.8,color='k', label='平滑曲线')

                #     # Func to draw line segment
                #     def newline(p1, p2, color='black'):
                #         ax = plt.gca()
                #         l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='r',lw=.7)
                #         ax.add_line(l)
                #         return l

                #     ax.plot_date(temp['time'],temp['smooth'],fmt='-',lw=.5,color='k')

                #     # Points
                #     ax.plot_date(temp['time'],temp[p],fmt='.',markersize=1, color='k', alpha=0.7)

                #     ax.plot_date(temp['time'].loc[rr],temp[p].loc[rr],fmt='.',markersize=5,label='原始数据', color='k', alpha=0.7)
                #     ax.plot_date(temp['time'].loc[rr],temp['x'].loc[rr],fmt='.',markersize=5,label='修复数据', color='r', alpha=0.7)

                #     # Line Segments
                #     for i, p1, p2 in zip(temp['time'].loc[rr], temp[p].loc[rr], temp['x'].loc[rr]):
                #         newline([i, p1], [i,p2])

                #     locator = mpl.dates.AutoDateLocator()
                #     formatter = mpl.dates.ConciseDateFormatter(locator)
                #     ax.xaxis.set_major_locator(locator)
                #     ax.xaxis.set_major_formatter(formatter)

                #     plt.legend()
                #     plt.xlabel('时间')
                #     plt.ylabel('数值')
                #     plt.savefig('images/%s/%s.png'%(itype, allcols[i]),dpi=600,bbox_inches='tight')

