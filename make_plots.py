import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import seaborn as sns

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Provide dataset name"
    _, dataset = sys.argv[1]
    acc_plot_title = "{} Dataset Performace".format(dataset)
    time_plot_title = "{} Dataset Runnig Time".format(dataset)

    # plt.rc('font', family='serif')
    # plt.rc('text', usetex=True)
    # use_files = {"sync_disc_10_5000_jtj9r9y9"}

    df = None
    dataset_dir = 'results/{}'.format(dataset)
    with open("{}".format(dataset_dir), 'r') as f:
        df = pd.read_csv(f, names=['eps', 'acc', 'seconds'])

    print(df.head())
    # sys.exit(1)
    # df["N"] = pd.to_numeric(df["N"])
    # df["dim"] = pd.to_numeric(df["dim"])
    df["eps"] = pd.to_numeric(df["eps"])
    df["acc"] = pd.to_numeric(df["acc"])
    df["time"] = pd.to_numeric(df["time"])

    title = dataset
    print(df.head())
    spread = 0.01
    plt.title(acc_plot_title)

    acc_df = df.groupby(['eps'])['acc'].mean() ## xcol is usually the epsilon
    std_df = df.groupby(['eps'])['acc'].std() ## xcol is usually the epsilon
    cnt_df = df.groupby(['eps'])['acc'].count()
    ## get std
    x = acc_df.index.values
    y = acc_df.values
    yerr = std_df.values

    plt.xticks(x.round(1))
    plt.xlabel("Epsilon")
    plt.ylabel('Accuracy')

    offset = spread * (i-1.0)
    plt.errorbar(x+offset, y, yerr=yerr, label="%s" % algo_name, fmt="-o", elinewidth=1.0, capsize=2.0, alpha=1.0, linewidth=2)

        # plt.plot(x, y,'-o', label="%s" % algo)
    legend = plt.legend(loc=4, framealpha=0.5)
    frame = legend.get_frame()
    # frame.set_facecolor('0.9')
    # frame.set_edgecolor('0.9')

    # dims = ','.join(str(dim) for dim in dim_schedule)
    # name = ','.join(algo.__name__ for algo in algo_list)
    # print('results/{}.png'.format(title))
    plt.savefig('plots/{}.png'.format(dataset))
    # plt.show()

    ## plot seconds
    # plt.clf()
    # plt.title(time_plot_title)
    # max_val = df['time'].max()
    # time_unit = 'Seconds' if max_val < 10*60 else 'Minutes'
    # max_val = max_val if time_unit == 'Seconds' else max_val / 60
    # b = min(int(max_val)+1, 20)
    # b = max(b, 10)
    #
    # # plt.title("running time ({})".format(time_unit))
    # data = []
    # lbls = []
    # for algo in algo_list:
    #     if algo[:2] == "OP":algo_name="OPDisc"
    #     if algo[:4] == "RSPM":algo_name="RSPM"
    #     if algo[:5] == "train":algo_name="Private LG"
    #     temp_df = df[df['algorithm'] == algo]
    #     y = temp_df["time"].values
    #     if time_unit == 'minutes':
    #         y= y/60
    #     data.append(y)
    #     lbls.append("%s" % algo_name)
    #
    # ax = sns.barplot(
    #     x="Epsilon", y="Running Time", hue="inference",
    #     order=[],
    #     hue_order=["oracle", "bayesian"],
    #     data=data)
    # plt.hist(data, bins=b, label=lbls)
    #
    # plt.xlabel(time_unit)
    # plt.legend()
    # plt.grid(True, axis='y')
    # plt.savefig('results/{}_time.png'.format(title))
