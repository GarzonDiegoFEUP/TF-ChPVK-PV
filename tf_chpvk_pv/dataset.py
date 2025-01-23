from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, RANDOM_SEED

app = typer.Typer()


@app.command()
def main():
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    create_dataset()
    train_test_split()


def create_dataset(input_path: Path = RAW_DATA_DIR / "shuffled_dataset_chalcogenide_pvk.csv",
                   output_path: Path = PROCESSED_DATA_DIR / "chpvk_dataset.csv",):
    
    logger.info("Processing dataset...")
    #load data
    df = pd.read_csv(input_path, index_col=0)

    #add rX_rB_ratio feature to the dataframe
    rX_rB_ratio="rB_rX_ratio**-1"
    df.eval('rX_rB_ratio =' + rX_rB_ratio, inplace=True)

    df.drop(index=[240,445], inplace=True)
    df.index = df.material

    df.drop(index=df[df.X == 'O'].index, inplace=True)

    df.drop(columns=['elements', 'material', 'A', 'B', 'X'], inplace=True)

    df['chi_AX_ratio'] = df['delta_chi_AX'] / df['delta_chi_AO']
    df['chi_BX_ratio'] = df['delta_chi_BX'] / df['delta_chi_BO']

    df['log_rA_rB_ratio'] = np.log(df['rA_rB_ratio'])

    df.to_csv(output_path)

    logger.success("Processing dataset complete.")


def train_test_split(ratio_splitting=0.8, 
                     input_path: Path = PROCESSED_DATA_DIR / "chpvk_dataset.csv",
                     output_train_path: Path = PROCESSED_DATA_DIR / "chpvk_train_dataset.csv",
                     output_test_path: Path = PROCESSED_DATA_DIR / "chpvk_test_dataset.csv"):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Creating train and test dataset...")
    # -----------------------------------------

    df = pd.read_csv(input_path, index_col=0)
    
    #train and test dataset size
    size = len(df)
    size_train = int(size * ratio_splitting)
    size_test = size - size_train

    # make lists of random numbers for train and test sets 
    inds = np.arange(size)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(inds)
    task_sizes_train = [size_train]
    task_sizes_test = [size_test]
    test_inds = [int(ii) for ii in np.sort(inds[:task_sizes_test[0]])]
    train_inds = [int(ii) for ii in np.sort(inds[task_sizes_test[0]:])]

    # test data frame
    index_id_test=df.index[test_inds]
    df_units_columns= df.columns.values[:]
    test_vals = df.loc[index_id_test, :].values
    test_df= pd.DataFrame(index=index_id_test,data=np.array(test_vals),columns=df_units_columns)

    # train data frame
    index_id_train=df.index[train_inds]
    train_vals = df.loc[index_id_train,:].values
    train_df= pd.DataFrame(index=index_id_train,data=np.array(train_vals),columns=df_units_columns)

    #save dataframes
    test_df.to_csv(output_test_path)
    train_df.to_csv(output_train_path)
    
    #save indices
    np.save(INTERIM_DATA_DIR / "test_inds.npy", test_inds)
    np.save(INTERIM_DATA_DIR / "train_inds.npy", train_inds)

    logger.success("Creating train and test dataset complete.")
    
    return train_df, test_df


if __name__ == "__main__":
    app()
