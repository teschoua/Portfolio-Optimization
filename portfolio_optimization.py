import pandas as pd
from pathlib import Path
import os
from os import walk
import datetime as dt
import math

import pickle as pkl


path_folder = Path(os.getcwd())

# dict_annualized_returns = {}
#
# for (_, _, filenames) in walk(path_folder / 'Dataset/ETF/'):
#     for filename in filenames:
#
#         if filename == '.DS_Store.txt':
#             continue
#
#         else:
#             df = pd.read_csv(path_folder / f'Dataset/ETF/{filename}', sep="\t", index_col='date', parse_dates=True)
#             df = df.drop(columns=['Unnamed: 7'])
#             df = df.rename(columns={'ouv': 'open',
#                                     'haut': 'high',
#                                     'bas': 'low',
#                                     'clot': 'close',
#                                     'vol': 'volume',
#                                     'dev': 'devise'
#                                     }
#                          )
#             # Calculate Annualized Returns
#             nb_months = math.floor((df.index[-1] - df.index[0]).days / 30)
#
#             # Keep ETF with historical data > 3 years
#             if nb_months > 3*12:
#                 total_return = (df.open[-1] - df.open[0]) / df.open[0]
#                 annualized_return = ((1 + total_return)**(12/nb_months))-1
#
#                 dict_annualized_returns[filename] = annualized_return
#
# with open(path_folder / 'Dataset/Pickles/dict_annualized_returns.pickle', 'wb') as handle:
#     pkl.dump(dict_annualized_returns, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(path_folder / 'Dataset/Pickles/dict_annualized_returns.pickle', 'rb') as handle:
    dict_annualized_returns = pkl.load(handle)

dict_annualized_returns_sorted = {k: v for k, v in sorted(dict_annualized_returns.items(), key=lambda item: item[1])}

df_etf_annualized_returns = pd.DataFrame({'name': dict_annualized_returns_sorted.keys(),
                                          'annualized_returns': dict_annualized_returns_sorted.values()
                                          })

df_etf_annualized_returns.set_index('name', inplace=True)
df_etf_annualized_returns = df_etf_annualized_returns.sort_values(by=['annualized_returns'], ascending=False)
print(df_etf_annualized_returns.head())



























