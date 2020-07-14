import pandas as pd
from sys import argv


df_one = pd.read_csv(argv[1])
df_two = pd.read_csv(argv[2])
#df_three = pd.read_csv(argv[3])
new_df = df_one.copy()
new_df.loc[:, 'healthy':] = 0.5 * df_one.loc[:, 'healthy':] + 0.5 * df_two.loc[:, 'healthy':]# + 0.3 * df_three.loc[:, 'healthy':]

new_df.to_csv('ensemble_952-975.csv', index=False)