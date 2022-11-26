import pandas as pd
from pathlib import Path
import os
from os import walk
import datetime as dt
import math
import numpy as np
import copy

import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, plotting

path_folder = Path(os.getcwd())

''' 
Need : 1. df with timestamp index
       2. Minimum nb of data
'''


def calculate_annualized_return(df, minimum_nb_months=0):
    nb_months = math.floor((df.index[-1] - df.index[0]).days / 30)
    # Keep ETF with historical data > minimum_nb_months
    if nb_months > minimum_nb_months:
        total_return = (df.iloc[-1].values[0] - df.iloc[0].values[0]) / df.iloc[0].values[0]
        annualized_return = ((1 + total_return) ** (12 / nb_months)) - 1
        return annualized_return
    else:
        raise Exception("Not enough data")


def save_dict_etfs_annualized_return():
    dict_annualized_returns = {}

    for (_, _, filenames) in walk(path_folder / 'Dataset/ETF/'):
        for filename in filenames:

            if filename == '.DS_Store.txt':
                continue

            else:
                df = pd.read_csv(path_folder / f'Dataset/ETF/{filename}', sep="\t", index_col='date', parse_dates=True)
                df = df.drop(columns=['Unnamed: 7'])
                df = df.rename(columns={'ouv': 'open',
                                        'haut': 'high',
                                        'bas': 'low',
                                        'clot': 'close',
                                        'vol': 'volume',
                                        'dev': 'devise'
                                        }
                               )
                # Calculate Annualized Returns, with historical data > 3 years = 3*12months
                annualized_return = calculate_annualized_return(df, 3 * 12)

                dict_annualized_returns[filename] = annualized_return

    with open(path_folder / 'Dataset/Pickles/dict_annualized_returns.pickle', 'wb') as handle:
        pkl.dump(dict_annualized_returns, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_dict_etfs_annualized_return():
    with open(path_folder / 'Dataset/Pickles/dict_annualized_returns.pickle', 'rb') as handle:
        dict_annualized_returns = pkl.load(handle)
    return dict_annualized_returns


def pick_best_etfs_annualized_return(dict_annualized_return, n_best):
    dict_annualized_return_sorted = {k: v for k, v in sorted(dict_annualized_return.items(), key=lambda item: item[1])}

    df_etf_annualized_returns = pd.DataFrame({'name': dict_annualized_return_sorted.keys(),
                                              'annualized_returns': dict_annualized_return_sorted.values()
                                              })
    df_etf_annualized_returns.set_index('name', inplace=True)

    filenames_etfs = df_etf_annualized_returns.sort_values(by=['annualized_returns'], ascending=False)[0:n_best].index

    return filenames_etfs


# Load ETFs historical data
def load_etfs_close_data(filenames):
    df_etfs = pd.DataFrame()

    for filename_etf in filenames:
        df_etf = pd.read_csv(path_folder / f'Dataset/ETF/{filename_etf}', sep="\t", index_col='date', parse_dates=True)
        df_etf = df_etf[['clot']]
        df_etf = df_etf.rename(columns={'clot': filename_etf.split('.')[0]})

        df_etfs = pd.concat([df_etfs, df_etf], axis=1)

    return df_etfs


def df_line_plot(df):
    sns.lineplot(data=df)
    plt.legend(loc='upper right')
    plt.show()


def calculate_sharpe_ratio_annualized(df, risk_free=0.01):
    # Calculate Volatility
    daily_returns = df.pct_change()
    volatility_annualized = daily_returns.close.std() * math.sqrt(252)

    # Calculate Annualized Return
    annualized_return = calculate_annualized_return(df)

    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free) / volatility_annualized

    return sharpe_ratio


def save_etfs_sharpe_ratio_annualized():
    dict_sharpe_ratio = {}
    for (_, _, filenames) in walk(path_folder / 'Dataset/ETF/'):
        for filename in filenames:

            if filename == '.DS_Store.txt':
                continue
            else:
                try:
                    df = pd.read_csv(path_folder / f'Dataset/ETF/{filename}', sep="\t", index_col='date',
                                     parse_dates=True)
                    df = df.drop(columns=['Unnamed: 7'])
                    df = df.rename(columns={'ouv': 'open',
                                            'haut': 'high',
                                            'bas': 'low',
                                            'clot': 'close',
                                            'vol': 'volume',
                                            'dev': 'devise'
                                            }
                                   )

                    sharpe_ratio = calculate_sharpe_ratio_annualized(df[['close']])

                    dict_sharpe_ratio[filename.split('.')[0]] = sharpe_ratio
                except Exception as e:
                    print(str(e) + f"for the file : {filename}")

    with open(path_folder / 'Dataset/Pickles/dict_etfs_sharp_ratio.pickle', 'wb') as handle:
        pkl.dump(dict_sharpe_ratio, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_dict_etf_sharpe_ratio():
    with open(path_folder / 'Dataset/Pickles/dict_etfs_sharp_ratio.pickle', 'rb') as handle:
        dict_etfs_sharpe_ratio = pkl.load(handle)
        df_etfs_sharpe_ratio = pd.DataFrame({'name': dict_etfs_sharpe_ratio.keys(),
                                             'sharpe_ratio': dict_etfs_sharpe_ratio.values()})
        df_etfs_sharpe_ratio = df_etfs_sharpe_ratio.set_index('name')

    return df_etfs_sharpe_ratio


def get_minimum_date_not_na(df):
    portfolio_etf_nan = df.notna()
    portfolio_etf_nan = pd.DataFrame(portfolio_etf_nan.apply(np.prod, axis=1))
    portfolio_etf_nan.columns = ['boolean']
    portfolio_etf_nan = portfolio_etf_nan.loc[portfolio_etf_nan['boolean'] == 1]
    return portfolio_etf_nan.index[0]


def plot_efficient_frontier(efficient_frontier):
    fig, ax = plt.subplots()

    ef_max_sharpe = copy.copy(efficient_frontier)
    ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    n_samples = 10000
    w = np.random.dirichlet(np.ones(efficient_frontier.n_assets), n_samples)
    rets = w.dot(efficient_frontier.expected_returns)
    stds = np.sqrt(np.diag(w @ efficient_frontier.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    plt.show()


def main():
    # Calculate annualized return for each ETF, and store each result in a dict with the filename_ETF as key
    # save_dict_etfs_annualized_return()

    # Load dict_etfs_annualized_return
    dict_annualized_return = load_dict_etfs_annualized_return()

    # Pick 10-th best annualized return ETFs
    filenames_etfs = pick_best_etfs_annualized_return(dict_annualized_return, 10)

    # Load ETFs Close Data
    df_etfs = load_etfs_close_data(filenames_etfs)

    # 1. Calculate sharpe ratio for each etf in the Dataset
    # save_etfs_sharpe_ratio_annualized()

    # 2. Load dict sharpe ratio & Converts to DF
    df_etfs_sharpe_ratio = load_dict_etf_sharpe_ratio()

    # 3. Portfolio ETFs (pick 5th best sharpe ratio ETFs and Load them with their filename)
    portfolio_etf_names = list(df_etfs_sharpe_ratio.sort_values(by='sharpe_ratio', ascending=False)[0:5].index)
    portfolio_etf_names = [name + '.txt' for name in portfolio_etf_names]
    portfolio_etf = load_etfs_close_data(portfolio_etf_names)

    # Drop AMUNDIETFPEASP500UCITSETFEUR_FR0013412285 column (Not enough values)
    portfolio_etf.drop(columns=['AMUNDIETFPEASP500UCITSETFEUR_FR0013412285'], inplace=True)
    # Drop NA
    portfolio_etf.dropna(inplace=True)

    # 4. Sharpe Ratio & Efficient Frontier
    mu = expected_returns.mean_historical_return(portfolio_etf)
    sigma = risk_models.sample_cov(portfolio_etf)

    ef = EfficientFrontier(mu, sigma)

    plot_efficient_frontier(ef)

    print(ef.n_assets)








if __name__ == "__main__":
    main()
