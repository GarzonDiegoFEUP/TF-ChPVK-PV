from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np


from config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    
    run_SISSO_model()
    
    # -----------------------------------------

def run_SISSO_model(input_path: Path = PROCESSED_DATA_DIR / "chpvk_dataset.csv",
                    output_path: Path = INTERIM_DATA_DIR / "features_sisso.csv",):
    
    # Load the dataset
    df = pd.read_csv(input_path)

    df_units=df.rename(columns={ 'rA':'rA (AA)',
                                 'rB':'rB (AA)', 
                                 'rX':'rX (AA)',
                                 'nA':'nA (Unitless)',
                                 'nB':'nB (Unitless)',
                                 'nX':'nX (Unitless)',
                                 'rA_rB_ratio':'rA_rB_ratio (Unitless)',
                                 'rA_rX_ratio':'rA_rX_ratio (Unitless)',
                                 'rB_rX_ratio':'rB_rX_ratio (Unitless)',
                                 'rS_A':'rS_A (AA)',
                                 'rP_A':'rP_A (AA)',
                                 'Z_A':'Z_A (elem_charge)',
                                 'HOMO_A':'HOMO_A (eV)',
                                 'LUMO_A':'LUMO_A (eV)',
                                 'EA_A':'EA_A (eV)',
                                 'IP_A':'IP_A (eV)',
                                 'rS_B':'rS_B (AA)',
                                 'rP_B':'rP_B (AA)',
                                 'Z_B':'Z_B (elem_charge)',
                                 'HOMO_B':'HOMO_B (eV)',
                                 'LUMO_B':'LUMO_B (eV)',
                                 'EA_B':'EA_B (eV)',
                                 'IP_B':'IP_B (eV)',
                                 'rS_X':'rS_X (AA)',
                                 'rP_X':'rP_X (AA)',
                                 'Z_X':'Z_X (elem_charge)',
                                 'HOMO_X':'HOMO_X (eV)',
                                 'LUMO_X':'LUMO_X (eV)',
                                 'EA_X':'EA_X (eV)',
                                 'IP_X':'IP_X (eV)',
                                 'rX_rB_ratio':'rX_rB_ratio (Unitless)',
                                 'delta_chi_BX':'delta_chi_BX (Unitless)',
                                 'delta_chi_AO':'delta_chi_AO (Unitless)',
                                 'delta_chi_BO':'delta_chi_BO (Unitless)',
                                })
    
    if output_path.is_file():
        logger.success("SISSO Features were already generated.")
    else:
        logger.info("Generating SISSO features from dataset...")
        inputs = createInputs(df_units)
        create_features_SISSO(df_units, inputs)
        logger.success("SISSO Features generation complete.")

def createInputs(df, 
                 cols_path: Path = RAW_DATA_DIR / "cols.csv",
                 ops_path: Path = RAW_DATA_DIR / "ops.csv", 
                 train_inds_path: Path = INTERIM_DATA_DIR / "train_inds.npy",
                 test_inds_path: Path = INTERIM_DATA_DIR / "test_inds.npy"):
    """
    Get the units from the dataset
    """

    from sissopp import Inputs, Unit, FeatureNode

    train_inds = np.load(train_inds_path)
    test_inds = np.load(test_inds_path)

    cols = np.loadtxt(cols_path, 'str', delimiter=',').flatten()
    ops = np.loadtxt(ops_path, 'str', delimiter=',').flatten()

    inputs = Inputs()
    inputs.allowed_ops = ops
    inputs.n_sis_select = 3000 
    inputs.max_rung = 3
    inputs.n_dim = 4
    inputs.calc_type = "classification"
    inputs.n_residual = 200
    inputs.task_names = ["all_mats"]
    inputs.task_sizes_train = [len(train_inds)]
    inputs.task_sizes_test = [len(test_inds)]

    inputs.leave_out_inds = list(test_inds)


    # exp_label values according to the previous list of random number for train and test sets
    inputs.prop_train = df.loc[df.index[train_inds], "exp_label"].values
    inputs.prop_test = df.loc[df.index[test_inds], "exp_label"].values
    inputs.prop_label = "exp_label"
    inputs.prop_unit = Unit("eV")
    ### inputs.global_param_optfalse = True  
    inputs.fix_intercept= True 

    # name of materials according to the previous list of random number for train and test sets
    inputs.sample_ids_train = df.index[train_inds].tolist()
    inputs.sample_ids_test = df.index[test_inds].tolist()


    #primary features - deletes the unit part of the features and then uses node to make the input
    phi_0 = []
    for cc, col in enumerate(cols):
        expr = col.split("(")[0].strip()
        if len(col.split("(")) == 2:
            unit = Unit(col.split("(")[1].split(")")[0].strip())
        else:
            unit = Unit()    
        phi_0.append(FeatureNode(cc, expr, df.loc[df.index[train_inds], col].tolist(), df.loc[df.index[test_inds], col].tolist(), unit))

    inputs.phi_0 = phi_0

    return inputs

def create_features_SISSO(df_units, inputs,
                          output_path: Path = INTERIM_DATA_DIR / "features_sisso.csv",
                          train_inds_path: Path = INTERIM_DATA_DIR / "train_inds.npy"):
    """
    Create SISSO features from the dataset
    """

    train_inds = np.load(train_inds_path)

    #Perform SISSO++ to create a feature space and select the features

    from sissopp import FeatureSpace
    feature_space = FeatureSpace(inputs)

    feature_space.sis(df_units.loc[df_units.index[train_inds], "exp_label"].values)

    features=[feat.expr for feat in feature_space.phi_selected]      # features
    vals = [feat.value for feat in feature_space.phi_selected]       # the values of the features

    vals=np.transpose(vals)                                          # transpose of the matrix to make it compatible with the indices
    index_id=df_units.index[train_inds]                              # indices of materials 

    feature_df=pd.DataFrame(index=index_id,data=np.array(vals),columns=features)   # make a data frame of the features and values of the features for different materials
    feature_df.to_csv(output_path)

if __name__ == "__main__":
    app()

