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


def platt_scaling_plot(t = 't_sisso', train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
                        output_path: Path = FIGURES_DIR / "platt_scaling_plot.png"):
    
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    threshold_t_sisso = tolerance_factor_dict[t][1]

    

    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path)
    
    if 'p_' + t in train_df.columns and 'p_' + t in test_df.columns:
        logger.info("Generating Platt Scaling plot from data...")
        plt.figure(figsize=(8,8))
        plot1=sns.scatterplot(x=t, y='p_' + t, data=concat,hue='exp_label', style='dataset',
                    palette=['red','blue'], markers=['s','o'], s=80)
        
        #Set x lims
        import numpy as np
        x_lims = [min([np.min(concat[t])-0.1,threshold_t_sisso-0.5]), max([np.max(concat[t])+0.1,threshold_t_sisso+0.5])]

        plot1.set_xlabel("$t_{sisso}$", fontsize=20)
        plot1.set_ylabel("$P(t_{sisso})$", fontsize=20)
        plot1.tick_params(labelsize=20)
        plt.xlim(x_lims[0], x_lims[1])
        plt.axvline(tolerance_factor_dict[t][1])
        plt.axhline(0.5,linestyle='--')
        plt.savefig(output_path, dpi=600)

        logger.success("Plot generation complete.")

    else:
        logger.error("Platt Scaling plot cannot be generated as the required columns are not present in the dataframes.")



def platt_scaling_plot_plotly(t='t_sisso', 
                        train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
                        output_path: Path = FIGURES_DIR / "platt_scaling_plot_plotly.html"):  # Changed to .html for interactivity
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Load train and test datasets
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    # Load tolerance factor dictionary
    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    threshold_t_sisso = tolerance_factor_dict[t][1]

    # Combine datasets
    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path, index=False)

    # Check if required columns exist
    if 'p_' + t in train_df.columns and 'p_' + t in test_df.columns:
        logger.info("Generating Platt Scaling plot from data...")

        # Create scatter plot using Plotly
        fig = px.scatter(
            concat, x=t, y='p_' + t, 
            color='exp_label', symbol='dataset',
            color_discrete_map={'Stable': 'red', 'Unstable': 'blue'},  # Adjust colors
            symbol_map={'train': 'square', 'test': 'circle'},  # Adjust markers
            size_max=10, 
            labels={t: r'$t_\text{sisso}$', 'p_' + t: r'$P(t_\text{sisso})$'},
            hover_data=['dataset']
        )

        # Add threshold vertical line
        fig.add_shape(
            type="line", x0=threshold_t_sisso, x1=threshold_t_sisso, y0=0, y1=1,
            line=dict(color="black", width=2, dash="dash")
        )

        # Add horizontal line at y=0.5
        fig.add_shape(
            type="line", x0=min(concat[t]), x1=max(concat[t]), y0=0.5, y1=0.5,
            line=dict(color="gray", width=2, dash="dot")
        )

        # Update layout
        fig.update_layout(
            title="Platt Scaling Plot",
            xaxis_title=r'$t_\text{sisso}$',
            yaxis_title='$P(t_\text{sisso})$',
            template="plotly_white",
            legend_title="Label"
        )

        # Save plot
        fig.write_html(output_path)  # Saves as interactive HTML
        logger.success(f"Plot saved as {output_path}")

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


def graph_periodic_table(stable_candidates_t_sisso, t='t_sisso', save_plot=True):
    from pymatviz import count_elements,  ptable_heatmap_plotly, ptable_heatmap
    import matplotlib.pyplot as plt
    import re

    element_counts = count_elements([re.sub(r'\d+', '', x) for x in stable_candidates_t_sisso])

    # Plot the periodic table heatmap
    ptable_heatmap(element_counts, log=False, heat_mode='value')#, show_values=True)
    #fig.update_layout(title=dict(text="<b>Elements in the chemical space</b>", x=0.36, y=0.9))
    if save_plot:
      txt_title = "periodic_table_heatmap_" + t + ".png"
      plt.savefig(FIGURES_DIR / txt_title, dpi=600)

    plt.show()


def spider_plot(df, title):

    # Libraries
    import matplotlib.pyplot as plt
    import pandas as pd
    from math import pi


    # ------- PART 1: Create background

    # number of variable
    categories=list(df.drop(columns=['group']))
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(15,15))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6, 0.8], ["0.2","0.4","0.6", "0.8"], color="grey", size=20)
    plt.ylim(0,1)


    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values=df.loc['S'].drop(['group']).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="S")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values=df.loc['Se'].drop(['group']).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Se")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Ind3
    values=df.loc['hal'].drop(['group']).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Hal.")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), prop={'size': 35})
    
    #change font_size
    #ax.tick_params(labelsize=5)
    
    #save the graph

    txt_title = 'radar plot - ' + title + '.png'

    plt.savefig(FIGURES_DIR / txt_title, bbox_inches='tight')


def plot_tau_star_histogram(threshold, df):
  
  import matplotlib.pyplot as plt
  import seaborn as sns

  fig = plt.figure(figsize=(8, 6))

  df_ = df.copy()
  df_.loc[df_['exp_label'] == 1, 'exp_label_'] = 'Perovskite'
  df_.loc[df_['exp_label'] == 0, 'exp_label_'] = 'Nonperovskite'


  sns.set_context('talk')
  ax = sns.histplot(data=df_, x='tau*', hue='exp_label_',
                    multiple='dodge', element='bars', bins=25,
                    hue_order=['Perovskite', 'Nonperovskite'])

  # Add axvspan calls with labels
  ax.axvspan(xmin=threshold, xmax=1.6, color='red', alpha=0.15, label='$\\tau$*' + f' > {threshold}')
  ax.axvspan(xmin=0, xmax=threshold, color='green', alpha=0.15, label= '$\\tau$*' + f' < {threshold}')

  # Get all handles and labels from the axis. This should include both histplot and axvspan.
  if ax.legend_ is not None:
      ax.legend_.set_title(None)

  # Create a unified legend from the collected handles and labels, without a title.
  # This will overwrite any default legend created by seaborn.
  #ax.legend(handles=handles, labels=labels, title=None)

  plt.xlim([0, 1.6])
  plt.xlabel('$\\tau$*')
  plt.ylabel('Counts')
  plt.tight_layout()

  plt.savefig(FIGURES_DIR / 'tau_star_histogram.png', dpi=600, bbox_inches='tight')

  plt.show()

def plot_t_star_histogram(thresholds, df):

    fig = plt.figure(figsize=(8, 6))

    df_ = df.copy()
    df_.loc[df_['exp_label'] == 1, 'exp_label_'] = 'Perovskite'
    df_.loc[df_['exp_label'] == 0, 'exp_label_'] = 'Nonperovskite'


    sns.set_context('talk')
    ax = sns.histplot(data=df_, x='t*', hue='exp_label_',
                        multiple='dodge', element='bars', bins=25,
                        hue_order=['Perovskite', 'Nonperovskite']
                        )

    # Add axvspan calls with labels
    ax.axvspan(xmin=0.3, xmax=thresholds[0], color='red', alpha=0.15, )
    ax.axvspan(xmin=thresholds[1], xmax=2.2, color='red', alpha=0.15,)

    ax.axvspan(xmin=thresholds[0], xmax=thresholds[1], color='green', alpha=0.15, )

    # Get all handles and labels from the axis. This should include both histplot and axvspan.
    if ax.legend_ is not None:
        ax.legend_.set_title(None)

    # Create a unified legend from the collected handles and labels, without a title.
    # This will overwrite any default legend created by seaborn.
    #ax.legend(handles=handles, labels=labels, title=None)

    plt.xlim([0.3, 2.2])
    plt.xlabel('t*')
    plt.ylabel('Counts')
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / 't_star_histogram.png', dpi=600, bbox_inches='tight')

    plt.show()

def plot_t_star_vs_p_t_sisso(df, thresholds):
    plt.figure(figsize=(8, 6))

    # Create a copy and map exp_label for better legend labels
    df_plot = df.copy()
    df_plot['exp_label_'] = df_plot['exp_label'].map({0: 'Nonperovskite', 1: 'Perovskite'})

    markers = {"Nonperovskite": "X", "Perovskite": "o"}

    ax = sns.scatterplot(
        data=df_plot,
        x='t*',
        y='p_tau*',
        hue='exp_label_',
        style='exp_label_',
        s=100, # size of the points
        alpha=1, # transparency
        hue_order=['Perovskite', 'Nonperovskite'],
        markers=markers,
    )

    ax.axvspan(xmin=0.3, xmax=thresholds[0], color='red', alpha=0.15, )
    ax.axvspan(xmin=thresholds[1], xmax=2.2, color='red', alpha=0.15,)

    ax.axvspan(xmin=thresholds[0], xmax=thresholds[1], color='green', alpha=0.15, )

    ax.set_xlabel('t*')
    ax.set_ylabel('P($\\tau$*)')

    # Remove title from legend
    if ax.legend_ is not None:
        ax.legend_.set_title(None)
        sns.move_legend(ax, loc='upper right')

    plt.xlim([0.3, 2.2])
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / 'P_tau_t_star_scatter.png', dpi=600, bbox_inches='tight')

    plt.show()

def colormap_radii(df, exp_df, clf_proba=None):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if clf_proba is None:
        from tf_chpvk_pv.modeling.train import train_platt_scaling

        train_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_train_dataset.csv")
        test_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_test_dataset.csv")
        
        with open(INTERIM_DATA_DIR / "tolerance_factor_classifiers.pkl", 'rb') as file:
            clfs = pickle.load(file)

        train_df, test_df, clf_proba = train_platt_scaling(train_df, test_df, clfs['t_sisso'])

    rA_range = [df.rA.min()-10, df.rA.max()+10]
    rB_range = [df.rB.min()-10, df.rB.max()+10]
    rX_range = [df.rX.min(), df.rX.max()]

    def calculate_t_sisso(rA, rB, rX):
        #(|((rA_rX_ratio + rB_rX_ratio) + (|rB_rX_ratio - log_rA_rB_ratio|)) - (rA_rX_ratio**3)|)
        #return np.abs( (np.abs( (rA/rX * rB/rX) - (np.exp(rB/rX)) )) - (np.abs( np.abs(rB/rX - np.log(rA/rB)) - (rA/rX) ) ) )
        return np.abs( ( (rA/rX + rB/rX) + (np.abs(rB/rX - np.log(rA/rB)) )) - (rA/rX)**3 )

    #All the radii in pm
    rA = np.linspace(rA_range[0], rA_range[1], 1000)
    rB = np.linspace(rB_range[0], rB_range[1], 1000)

    xv, yv = np.meshgrid(rA, rB)

    t_sisso_S = calculate_t_sisso(xv, yv, rX_range[0])
    t_sisso_Se = calculate_t_sisso(xv, yv, rX_range[1])

    p_t_sisso_S = clf_proba.predict_proba(t_sisso_S.reshape(-1,1))[:,1]
    p_t_sisso_Se = clf_proba.predict_proba(t_sisso_Se.reshape(-1,1))[:,1]

    sns.set_context('talk')

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2) # Increased wspace

    color_map = 'coolwarm_r'

    rA_range = [110, 180]
    rB_range = [50, 120]

    # Plot for S anion (rX_range[0])
    scatter_s = axes[0].scatter(xv, yv, c=p_t_sisso_S, cmap=color_map, vmin=0, vmax=1)
    axes[0].set_xlabel('$r_A$ (pm)')
    axes[0].set_ylabel('$r_B$ (pm)')
    axes[0].set_xlim(rA_range)
    axes[0].set_ylim(rB_range)
    axes[1].tick_params(left=True, right=True)
    axes[0].set_title('ABS$_3$ compounds')

    # Plot for Se anion (rX_range[1])
    axes[1].scatter(xv, yv, c=p_t_sisso_Se, cmap=color_map, vmin=0, vmax=1)
    axes[1].set_xlabel('$r_A$ (pm)')
    axes[1].set_xlim(rA_range)
    axes[1].set_ylim(rB_range)
    axes[1].set_yticklabels([])
    axes[1].tick_params(left=True, right=True)
    axes[1].set_title('ABSe$_3$ compounds')

    # Add a single color bar for both subplots
    fig.colorbar(scatter_s, ax=axes.ravel().tolist(), label='$P(\\tau*)$')

    #remove non-chalcogenides from exp_df
    exp_df = exp_df[exp_df.rX.isin(rX_range)]
    #delete rX = 196 pm (Br) from exp_df
    exp_df = exp_df[exp_df.rX != 196]

    #Add experimentally observed compounds
    for idx in exp_df.index:
        d = exp_df.loc[idx]
        rA_ = d.rA
        rB_ = d.rB
        X = d.rX
        if d.exp_label == 1:
            if X == rX_range[0]:
                axes[0].scatter(rA_, rB_, marker='s', color='black', s=50)
            elif X == rX_range[1]:
                axes[1].scatter(rA_, rB_, marker='s', color='black', s=50)
        else:
            if X == rX_range[0]:
                axes[0].scatter(rA_, rB_, marker='^', color='black', s=50)
            elif X == rX_range[1]:
                axes[1].scatter(rA_, rB_, marker='^', color='black', s=50)

    plt.savefig(FIGURES_DIR / 'p_t_sisso color_map radii.png', dpi=600)
    plt.show()


def confusion_matrix_plot(df, test=True):


    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set_context('talk')

    y_true = df['exp_label'].values

    y_pred = np.zeros_like(y_true)

    y_pred[df['p_t_sisso'] >= 0.5] = 1

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if test:
        plt.title('Confusion Matrix - Test Set')
    else:
        plt.title('Confusion Matrix - Train Set')

    plt.savefig(FIGURES_DIR / f'confusion_matrix_{"test" if test else "train"}.png', dpi=600)
    plt.show()



def plot_matrix(df_out, df_crystal, anion='S', parameter='Eg', clf_proba=None):


    import matplotlib.pyplot as plt
    from matplotlib.markers import MarkerStyle
    from tf_chpvk_pv.config import FIGURES_DIR

    import numpy as np
    import seaborn as sns

    element_to_number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
    "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
    }

    if parameter == 'p_t_sisso' and clf_proba is None:
        from tf_chpvk_pv.modeling.train import train_platt_scaling

        train_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_train_dataset.csv")
        test_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_test_dataset.csv")
        
        with open(INTERIM_DATA_DIR / "tolerance_factor_classifiers.pkl", 'rb') as file:
            clfs = pickle.load(file)

        train_df, test_df, clf_proba = train_platt_scaling(train_df, test_df, clfs['t_sisso'])
        df_out['p_t_sisso'] = clf_proba.predict_proba(df_out['t_sisso'].values.reshape(-1,1))[:,1]     

    df_sisso = df_out[df_out['formula'].isin(df_crystal['formula'])]
    
    df_out = df_out[df_out['X'] == anion].copy()
    if parameter in ['Eg']:
        df_out = df_out[df_out[parameter] > 0].copy()
    df_out['Z_A'] = df_out.A.map(element_to_number)
    df_out['Z_B'] = df_out.B.map(element_to_number)
    df_out.sort_values(by=['Z_A', 'Z_B'], inplace=True, ascending=[True, True])

    df_sisso = df_sisso[df_sisso['X'] == anion].copy()
    if parameter in ['Eg']:
        df_sisso = df_sisso[df_sisso[parameter] > 0].copy()
    df_sisso['Z_A'] = df_sisso.A.map(element_to_number)
    df_sisso['Z_B'] = df_sisso.B.map(element_to_number)
    df_sisso.sort_values(by=['Z_A', 'Z_B'], inplace=True, ascending=[True, True])


    #sns.set_theme(style="whitegrid")
    sns.set_context('talk')

    df_crab = df_sisso

    size_markers = 200

    if parameter == 'Eg':
        if anion == 'S':
            fsize = df_out.nunique()['A'] / 2.8
        elif anion == 'Se':
            fsize = df_out.nunique()['A'] / 2.6
    elif parameter == 'p_t_sisso':
        fsize = df_out.nunique()['A'] / 3

    fig, ax = plt.subplots(figsize=(fsize+3, fsize))


    

    if parameter == 'Eg':
        vmin_, vmax_ = 0.5, 3.5
        cmap_ = 'jet'
    elif parameter == 'p_t_sisso':
        vmin_, vmax_ = 0.0, 1.0
        cmap_ = 'coolwarm_r'
    else:
        vmin_, vmax_ = df_out[parameter].min(), df_out[parameter].max()

        
    im1 = ax.scatter(df_out.B, df_out.A, c=df_out[parameter], marker='s',
                vmin=vmin_, vmax=vmax_, cmap=cmap_, s=size_markers)
    
    ax.scatter(df_crab.B, df_crab.A, marker='s',
                edgecolor="black", color="None", s=size_markers)

    cbar1 = plt.colorbar(im1, ax=ax)
    if parameter == 'Eg':
        cbar1.set_label("Bandgap (eV) ", rotation=90)
    elif parameter == 'p_t_sisso':
        cbar1.set_label(r"$P(\tau*)$ ", rotation=90)


    plt.title('')

    # Tweak the figure to finalize
    ax.set(xlabel="Cation B", ylabel="Cation A", aspect="equal")

    plt.xticks(rotation=90)
    #plt.grid(color='None', linestyle='-')
    #ax.set_frame_on(False)
    ax.grid(False)

    if parameter == 'Eg':
        name_file = 'matrix_' + f'{parameter.lower()}_crabnet_' + anion + '.png'
    elif parameter == 'p_t_sisso':
        name_file = 'matrix_' + f'{parameter.lower()}_' + anion + '.png'

    plt.savefig(FIGURES_DIR / name_file, dpi=600, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    app()
