import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    df = pd.read_csv("C:/Users/User/gits/predictprotein1_solubility/wandb_linear_models.csv")
    df = df.loc[:, ['Name', 'acc', 'auc', 'bal_acc', "epoch", 'f1', 'loss', 'mcc', 'precision', 'recall', 'test_loss', 'val_loss']]
    df['model'] = df['Name'].str.split('-', n=1, expand=True)[0]
    df.drop('Name', axis=1, inplace=True)
    df = pd.melt(df, id_vars=['model'], var_name='metric', value_name='value')

    # print mean and standard deviation of metrics
    df_agg = df.groupby(["model", "metric"])["value"].agg("mean", "std")
    print(df_agg)
    # df_agg.to_csv("")

    # set colors for plotting

    colors = {"OneLayerLin": "royalblue",
              "TwoLayerLin64": "cornflowerblue",
              "TwoLayerLin128": "lightblue",
              "TwoLayerLin256": 'steelblue',
              "TwoLayerLin512": 'dodgerblue',
              "ThreeLayerLin128x64": 'skyblue'}

    # Set the size of the plot
    plt.figure(figsize=(8, 4))

    # plot with seaborn
    sns.barplot(df[~df['metric'].isin(['epoch', 'test_loss'])], x="metric", y="value", hue="model", errorbar='se',
                palette=colors)
    plt.show()

    # Set the size of the plot
    plt.figure(figsize=(3, 4))
    # plot number of epochs
    sns.barplot(df[df['metric'].isin(['epoch'])], x="metric", y="value", hue="model", errorbar='se',
                palette=colors)
    plt.show()

