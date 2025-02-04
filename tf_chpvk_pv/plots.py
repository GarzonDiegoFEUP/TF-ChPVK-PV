from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tf_chpvk_pv.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main():

    platt_scaling_plot()

    plot_t_sisso_tf("t")
    plot_t_sisso_tf("t_jess")
    plot_t_sisso_tf("tau")

    plot_p_t_sisso_tf("t")
    plot_p_t_sisso_tf("t_jess")
    plot_p_t_sisso_tf("tau")


def platt_scaling_plot(train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
                        output_path: Path = FIGURES_DIR / "platt_scaling_plot.png"):
    
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    threshold_t_sisso = tolerance_factor_dict["t_sisso"][1]

    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path)
    
    if 'p_t_sisso' in train_df.columns and 'p_t_sisso' in test_df.columns:
        logger.info("Generating Platt Scaling plot from data...")
        plt.figure(figsize=(8,8))
        plot1=sns.scatterplot(x='t_sisso', y='p_t_sisso', data=concat,hue='exp_label', style='dataset',
                    palette=['red','blue'], markers=['s','o'], s=80)
        plot1.set_xlabel("$t_{sisso}$", fontsize=20)
        plot1.set_ylabel("$P(t_{sisso})$", fontsize=20)
        plot1.tick_params(labelsize=20)
        plt.xlim(threshold_t_sisso-0.5, threshold_t_sisso+0.5)
        plt.axvline(tolerance_factor_dict["t_sisso"][1])
        plt.axhline(0.5,linestyle='--')
        plt.savefig(output_path)

        logger.success("Plot generation complete.")

    else:
        logger.error("Platt Scaling plot cannot be generated as the required columns are not present in the dataframes.")


def plot_t_sisso_tf(tf, train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
):
    
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    threshold_t_sisso = tolerance_factor_dict["t_sisso"][1]

    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path)

    dict_labels = {"t": "$t$", "t_sisso": "$t_{sisso}$", 't_jess': "$t_{jess}$", 'tau': r"$\tau$"}

    
    if "t_sisso" in concat.columns and tf in concat.columns:
        logger.info(f"Generating t_sisso as a function of {tf} plot from data...")
        plt.figure(figsize=(8,8))
        plot2=sns.scatterplot(x=tf, y='t_sisso', data=concat, hue='exp_label', style='dataset', 
                            palette=['red','blue'], markers=['s','o'], s=80)
        plot2.set_xlabel(dict_labels[tf], fontsize=20)
        plot2.set_ylabel(dict_labels['t_sisso'], fontsize=20)
        plot2.tick_params(labelsize=20)
        plt.ylim(threshold_t_sisso-4, threshold_t_sisso+4)
        tresholds = tolerance_factor_dict[tf][1]
        if isinstance(tresholds, list):
            plt.axvline(tresholds[0])
            plt.axvline(tresholds[1])
        else:
            plt.axvline(tresholds)
            plt.xlim(tresholds-4, tresholds+4)

        plt.axhline(threshold_t_sisso)
        name_figure = f"t_sisso as a function of {tf}.png"
        path_figure: Path = FIGURES_DIR / name_figure
        plt.savefig(path_figure)
        logger.success("Plot generation complete.")
    else:
        logger.error(f"t_sisso as a function of {tf} plot cannot be generated as the required columns are not present in the dataframes.")

def plot_p_t_sisso_tf(tf, train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
):
    
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path)

    dict_labels = {"t": "$t$", "p_t_sisso": "$P(t_{sisso})$", 't_jess': "$t_{jess}$", 'tau': r"$\tau$"}

    
    if "p_t_sisso" in concat.columns and tf in concat.columns:
        logger.info(f"Generating P(t_sisso) as a function of {tf} plot from data...")
        plt.figure(figsize=(8,8))
        plot2=sns.scatterplot(x=tf, y='p_t_sisso', data=concat, hue='exp_label', style='dataset', 
                            palette=['red','blue'], markers=['s','o'], s=80)
        plot2.set_xlabel(dict_labels[tf], fontsize=20)
        plot2.set_ylabel(dict_labels['p_t_sisso'], fontsize=20)
        plot2.tick_params(labelsize=20)
        tresholds = tolerance_factor_dict[tf][1]
        if isinstance(tresholds, list):
            plt.axvline(tresholds[0])
            plt.axvline(tresholds[1])
        else:
            plt.axvline(tresholds)
            plt.xlim(tresholds-4, tresholds+4)

        name_figure = f"P(t_sisso) as a function of {tf}.png"
        path_figure: Path = FIGURES_DIR / name_figure
        plt.savefig(path_figure)
        logger.success("Plot generation complete.")
    else:
        logger.error(f"P(t_sisso) as a function of {tf} plot cannot be generated as the required columns are not present in the dataframes.")


if __name__ == "__main__":
    app()
