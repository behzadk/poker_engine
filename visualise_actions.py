import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def action_heatmap():
    csv_path = "./output/player_actions/player_bvb_0_actions.csv"
    data = np.genfromtxt(csv_path, delimiter=',')

    num_q_values = 8

    state_columns = ['S_' + str(i) for i in range(np.shape(data)[1] - num_q_values)]
    column_names = state_columns
    column_names.extend(['Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5', 'Q_6', 'Q_7', 'Q_8'])
    # column_names.append('Q_2')




    data_df = pd.DataFrame(data, columns=column_names)
    # data_df = data_df.loc[data_df['S_6'] == 1]
    data_df = data_df[['S_0', 'S_1', 'S_2', 'Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5', 'Q_6', 'Q_7', 'Q_8']]
    raise_columns = ['Q_5', 'Q_6', 'Q_7', 'Q_8']

    data_df['max_raise'] = data_df.apply(lambda x: np.max(x[raise_columns]), axis=1)

    suited_df = data_df.loc[data_df['S_2'] == 1]
    unsuited_df = data_df.loc[data_df['S_2'] == 0]

    # suited_df = suited_df.groupby(['S_0', 'S_1'])['action'].mean().reset_index(name='mean_Q_ratio')
    # suited_df = suited_df.groupby(['S_1', 'S_2'])['Q_1'].mean().reset_index(name='mean_F_Q_1')

    # unsuited_df = unsuited_df.groupby(['S_0', 'S_1'])['action'].mean().reset_index(name='mean_Q_ratio')
    unsuited_df_max_raise = unsuited_df.groupby(['S_0', 'S_1'])['max_raise'].mean().reset_index(name='max_raise')
    unsuited_df_fold = unsuited_df.groupby(['S_0', 'S_1'])['Q_4'].mean().reset_index(name='Q_4')

    # unsuited_df = unsuited_df.groupby(['S_0', 'S_1'])['max_raise'].mean().reset_index(name='mean_F_Q_4')

    unsuited_df_max_raise['fold'] = unsuited_df_fold['Q_4'].values
    df = unsuited_df_max_raise
    print(df)

    df['raise_fold'] = df.apply(lambda x: (0 if x['max_raise'] > x['fold'] else 1), axis=1)

    heat_arr = np.zeros([13, 13])

    for idx, row in df.iterrows():
        x = int(row['S_0']) - 1
        y = int(row['S_1']) - 1

        print(x, y)

        val = row['raise_fold']

        heat_arr[x][y] = val
        heat_arr[y][x] = val

    sns.heatmap(heat_arr)
    plt.show()



if __name__ == "__main__":
    action_heatmap()
