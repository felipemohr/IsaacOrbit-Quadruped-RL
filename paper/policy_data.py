import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="Go2 environment.")
parser.add_argument("--csv_files_path", type=str, default=None, help="Path containing csv files.")
parser.add_argument("--output_plots_path", type=str, default=None, help="Path to save the plots.")
args_cli = parser.parse_args()

if args_cli.output_plots_path is not None:
    if not os.path.exists(args_cli.output_plots_path):
        os.makedirs(args_cli.output_plots_path)

avg_window_size = 10
files = os.listdir(args_cli.csv_files_path)
for file in files:
    filepath = os.path.join(args_cli.csv_files_path, file)
    print(f"Reading {filepath}")

    df = pd.read_csv(filepath)
    df_avg = df["Value"].rolling(window=10).mean()

    y_range = df_avg.max() - df_avg.min()
    ylim_min = df_avg.min() - 0.1 * y_range
    ylim_max = df_avg.max() + 0.1 * y_range

    plt.clf()
    plt.close("all")
    plt.figure(figsize=(8, 4))
    plt.title(file.split(".")[0], fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(ylim_min, ylim_max)
    plt.plot(df["Step"], df_avg, linewidth=2.5)
    plt.plot(df["Step"], df["Value"], alpha=0.3, linewidth=2.0)
    plt.grid(True, which="both", linestyle="--", linewidth=1.0)

    if args_cli.output_plots_path:
        save_plot_path = os.path.join(args_cli.output_plots_path, file.replace(".csv", ".png"))
        plt.savefig(save_plot_path, bbox_inches="tight", dpi=300)
    plt.show()