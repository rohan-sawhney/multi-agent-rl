#!/usr/bin/env python3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import argparse
import os


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CSV Plotter",
                                     description="Plots CSV files")
    parser.add_argument("filename",
                        type=str,
                        help="The filename of csv format")
    parser.add_argument("-o",
                        "--output_filename",
                        nargs="?",
                        type=str,
                        help="The filename to write the plot to")
    parser.add_argument("-s",
                        "--startidx",
                        nargs="?",
                        type=int,
                        default=0,
                        help="Where to start from")
    parser.add_argument("-e",
                        "--endidx",
                        nargs="?",
                        type=int,
                        default=-2,
                        help="Where to end")
    parser.add_argument("-r",
                        "--rolling",
                        nargs="?",
                        type=int,
                        default=0,
                        help="Rolling average size")
    parser.add_argument("--loss",
                        action="store_true")
    parser.add_argument("--reward",
                        action="store_true")
    parser.add_argument("--collisions",
                        action="store_true")
    parser.add_argument("--cum_reward",
                        action="store_true")
    parser.add_argument("--steps",
                        action="store_true")
    parser.add_argument("--error",
                        action="store_true")
    parser.add_argument("--perplexity",
                        action="store_true")
    parser.add_argument("--autoout",
                        action="store_true")
    parser.add_argument("--min",
                        action="store_true")

    args = parser.parse_args()
    df = pd.read_csv(args.filename, delimiter=",", header=0, index_col=0)

    selected = df.keys()
    if args.loss or args.error or args.perplexity or args.reward or \
            args.cum_reward or args.collisions or args.steps:
        selected = []
        if args.loss:
            selected.extend([k for k in df.keys() if "loss" in k])
        if args.reward:
            selected.extend([k for k in df.keys() if "reward" in k])
        if args.cum_reward:
            selected.extend([k for k in df.keys() if "cum_reward" in k])
        if args.collisions:
            selected.extend(
                [k for k in df.keys() if "collisions" in k])
        if args.steps:
            selected.extend([k for k in df.keys() if "steps" in k])
        if args.error:
            selected.append("validation_error")
            selected.append("train_error")
        if args.perplexity:
            selected.append("validation_perplexity")
            selected.append("train_perplexity")
    selected_df = df[selected][args.startidx:args.endidx]

    print("Mins\n{}".format(selected_df.min()))
    print("*" * 80)
    print("Means\n{}".format(selected_df.mean()))
    print("*" * 80)
    print("Varss\n{}".format(selected_df.var()))
    print("*" * 80)
    print("Maxs\n{}".format(selected_df.max()))
    print("*" * 80)

    ax = selected_df.plot()
    if args.rolling:
        window_size = args.rolling
        selected_df.rolling(window_size).mean().plot(
            color="k", style="--", ax=ax, legend=0)
    if args.min:
        idxs = selected_df.idxmin()
        for k in selected_df:
            x = idxs[k]
            y = selected_df[k][x]
            if k == "validation_error":
                y_test_error = df["test_error"][x]
                y_test_loss = df["test_loss"][x]
                y_validation_error = df["validation_error"][x]
                y_validation_loss = df["validation_loss"][x]
                y_train_error = df["train_error"][x]
                y_train_loss = df["train_loss"][x]
                print("Low validation error at x={}".format(x))
                print("Optimized test loss {}".format(y_test_loss))
                print("Optimized test error {}".format(y_test_error))
                print("Optimized validation loss {}".format(y_validation_loss))
                print("Optimized validation error {}".format(y_validation_error))
                print("Optimized train loss {}".format(y_train_loss))
                print("Optimized train error {}".format(y_train_error))
            if k == "validation_loss":
                y_test_error = df["test_error"][x]
                y_test_loss = df["test_loss"][x]
                y_validation_error = df["validation_error"][x]
                y_validation_loss = df["validation_loss"][x]
                y_train_error = df["train_error"][x]
                y_train_loss = df["train_loss"][x]
                print("Low validation loss at x={}".format(x))
                print("Optimized test loss {}".format(y_test_loss))
                print("Optimized test error {}".format(y_test_error))
                print("Optimized validation loss {}".format(y_validation_loss))
                print("Optimized validation error {}".format(y_validation_error))
                print("Optimized train loss {}".format(y_train_loss))
                print("Optimized train error {}".format(y_train_error))
            ax.scatter(x, y, s=40, facecolors="none", edgecolors="r")
    plt.xlabel("episodes")
    if args.loss and not args.error:
        plt.ylabel("loss")
    elif args.error and not args.loss:
        plt.ylabel("error")
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
    else:
        plt.ylabel("loss/error")
    #plt.title("NN results")
    if args.output_filename:
        plt.savefig(args.output_filename)
    elif args.autoout:
        plt.savefig(os.path.splitext(args.filename)[0] + ".pdf")
    else:
        plt.show()
