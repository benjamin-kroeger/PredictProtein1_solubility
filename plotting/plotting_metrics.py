import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Plotting metrics from predictions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data related arguments
    parser.add_argument('--baseline', type=str, required=False,
                        help='Path to the csv file for baseline model')
    parser.add_argument('--finetune', type=str, required=False,
                        help='Path to the csv file for finetune model')
    parser.add_argument('--linear', type=str, required=False,
                        help='Path to the csv file for linear model')
    parser.add_argument('--attention', type=str, required=False,
                        help='Path to the csv file for light attention model')

    # metrics
    parser.add_argument('--acc', action='store_true', default=False)
    parser.add_argument('--bal_acc', action='store_true', default=False)
    parser.add_argument('--f1', action='store_true', default=False)
    parser.add_argument('--recall', action='store_true', default=False)
    parser.add_argument('--precision', action='store_true', default=False)
    parser.add_argument('--auc', action='store_true', default=False)
    parser.add_argument('--mcc', action='store_true', default=False)
    parser.add_argument('--test_loss', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = init_parser()

    # parse metrics
    metrics = []
    if args.acc:
        metrics.append("acc")
    if args.bal_acc:
        metrics.append("bal_acc")
    if args.f1:
        metrics.append("f1")
    if args.recall:
        metrics.append("recall")
    if args.precision:
        metrics.append("precision")
    if args.auc:
        metrics.append("auc")
    if args.mcc:
        metrics.append("mcc")
    if args.test_loss:
        metrics.append("test_loss")

    # check if at least one metric is specified
    if len(metrics) == 0:
        raise Exception("At least one metric must be specified!")

    # create empty dfs
    baseline_df = pd.DataFrame()
    finetune_df = pd.DataFrame()
    linear_df = pd.DataFrame()
    attention_df = pd.DataFrame()

    # NetSolP results
    data = {
        'metric': ['acc', 'test_acc', 'mcc', 'precision', 'auc'],
        'mean': [0.7, 0.782, 0.402, 0.773, 0.760],
        'std': [0.02, 0, 0, 0, 0]
    }

    netsolp_df = pd.DataFrame(data)
    netsolp_df["model"] = "NetSolP"

    # check if at least one file was specified
    if not (args.baseline or args.finetune or args.linear or args.attention):
        raise Exception("At least one csv file must be specified.")

    # read the files
    if args.baseline:
        baseline_df = pd.read_csv(args.baseline)
        baseline_df["model"] = "baseline"
        baseline = True
    if args.finetune:
        finetune_df = pd.read_csv(args.finetune)
        finetune_df["model"] = "finetune"
        finetune = True
    if args.linear:
        linear_df = pd.read_csv(args.linear)
        linear_df["model"] = "linear"
        linear = True
    if args.attention:
        attention_df = pd.read_csv(args.attention)
        attention_df["model"] = "light-attention"

    # concatenate dataframes
    df = pd.concat([baseline_df, finetune_df, linear_df, attention_df])

    # print mean and standard deviation of metrics
    df_agg = df.groupby(["model", "metric"])["value"].agg(["mean", "std"])
    print(df_agg)
    print(netsolp_df)

    # Set the size of the plot
    plt.figure(figsize=(8, 4))

    # plot with seaborn
    netsolp_df["value"] = netsolp_df["mean"]
    df = pd.concat([df, netsolp_df])
    colors = {"baseline": (0.7, 0.7, 0.7),
              "finetune": "royalblue",
              "linear": "cornflowerblue",
              "light-attention": "skyblue",
              "NetSolP": "lightseagreen"}
    sns.barplot(df[df["metric"].isin(metrics)], x="metric", y="value", hue="model", errorbar='se',
                palette=colors)
    plt.legend(loc='lower left')
    plt.show()
