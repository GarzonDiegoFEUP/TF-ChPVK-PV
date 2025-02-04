from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np

from tf_chpvk_pv.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, RANDOM_SEED

app = typer.Typer()


@app.command()
def main():
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    create_dataset()
    train_test_split()

    elements_selection = ["Si", "Ge", "V", "Rh", "Ti", "Ru", "Mo", "Ta", "Nb",
                          "Sn", "Hf", "Zr", "Tb", "Pb", "Pt", "Ce", "U", "Ba",
                          "Eu", "Sr", "Ca", "Cd", "Cu", "Mg", "Zn", "Ge", "Fe",
                          "Nb", "La", "Pr", "Nd", "Yb", "Gd", "Sm", "Y", "Dy", "Ho",
                          "Er", "Tm", "Lu", "Sc", "Tl", "Bi",  "Pd", "Ni", "Co", "Ga",
                          "Al", "Cr", "In", "V", "Mn", "Tm"]

    generate_compositions(elements_selection)


def create_dataset(input_path: Path = RAW_DATA_DIR / "shuffled_dataset_chalcogenide_pvk.csv",
                   output_path: Path = PROCESSED_DATA_DIR / "chpvk_dataset.csv",
                   new_radii_path: Path = RAW_DATA_DIR / "Expanded_Shannon_Effective_Ionic_Radii.csv"):
    
    from pymatgen.core import Composition, periodic_table
    
    logger.info("Processing dataset...")
    #load data
    df = pd.read_csv(input_path, index_col=0)

    #correct radii values
    new_radii = pd.read_csv(new_radii_path) 

    for idx in df.index:
    
        nA = df.loc[idx, 'nA']
        nB = df.loc[idx, 'nB']
        nX = df.loc[idx, 'nX']

        A = df.loc[idx, 'A']
        B = df.loc[idx, 'B']
        X = df.loc[idx, 'X']

        """if nA > 0:
            nA_ = '+' + str(round(nA))
        else:
            nA_ = str(round(nA))

        if nB > 0:
            nB_ = '+' + str(round(nB))
        else:
            nB_ = str(round(nB))

        if nX > 0:
            nX_ = '+' + str(round(nX))
        else:
            nX_ = str(round(nX))"""

        Z_A = periodic_table.Element(A).Z
        Z_B = periodic_table.Element(B).Z
        Z_X = periodic_table.Element(X).Z

        #rA_ = radii.loc[(radii.ION  == A + ' ' + nA_ ) & (radii['Coord. #'] == 12), 'Ionic Radius'].values
        rA_ = new_radii.loc[(new_radii["Atomic Number"] == Z_A) & (new_radii["Oxidation State"] == nA) &(new_radii['Coordination Number'] == 12), 'Mean'].values
        
        #if np.shape(rA_)[0] == 0:
            #rA_ = radii.loc[(radii.ION  == A + nA_ ) & (radii['Coord. #'] == 12), 'Ionic Radius'].values

        if np.shape(rA_)[0] == 0:
            #rA_ = radii.loc[(radii.ION  == A + nA_ ) & (radii['Coord. #'] == 8), 'Ionic Radius'].values
            rA_ = new_radii.loc[(new_radii["Atomic Number"] == Z_A) & (new_radii["Oxidation State"] == nA) & (new_radii['Coordination Number'] == 8), 'Mean'].values

        if np.shape(rA_)[0] > 0:  
            df.loc[idx, 'rA'] = rA_

        #rB_ = radii.loc[(radii.ION  == B + ' ' + nB_ ) & (radii['Coord. #'] == 6), 'Ionic Radius'].values
        rB_ = new_radii.loc[(new_radii["Atomic Number"] == Z_B) & (new_radii["Oxidation State"] == nB) & (new_radii['Coordination Number'] == 6), 'Mean'].values

        #if np.shape(rB_)[0] == 0:
        #    rB_ = radii.loc[(radii.ION  == B + nB_ ) & (radii['Coord. #'] == 6), 'Ionic Radius'].values

        if np.shape(rB_)[0] > 0:  
            try:
                df.loc[idx, 'rB'] = rB_

            except:
                df.loc[idx, 'rB'] = rB_[0]

        #rX_ = radii.loc[radii.ION  == X + nX_, 'Ionic Radius'].values
        rX_ = new_radii.loc[(new_radii["Atomic Number"] == Z_X) & (new_radii["Oxidation State"] == nX) & (new_radii['Coordination Number'] == 6), 'Mean'].values
        df.loc[idx, 'rX'] = rX_


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

    return df


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

def generate_compositions(element_symbols, anions=["S", "Se"],
                          dict_tol_factors_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
                          output_path: Path = PROCESSED_DATA_DIR / "valid_new_compositions.csv",):
    """
    Generate valid compositions for perovskite materials.

    This function takes a list of element symbols and generates valid compositions
    for perovskite materials by combining unique pairs of cations with specified anions.
    The generated compositions are checked for validity and oxidation states are guessed.

    Parameters:
    - element_symbols (list): A list of element symbols representing the cations.
    - anions (list, optional): A list of anion symbols. Default is ["S", "Se"].

    Returns:
    - valid_compositions (list): A list of tuples containing the reduced formula and
      guessed oxidation states for each valid composition.

    Example usage:
    >>> element_symbols = ["Ba", "Ti", "O"]
    >>> generate_compositions(element_symbols)
    [('BaTiO3', {'Ba': 2, 'Ti': 4, 'O': -2})]

    """
    
    from pymatgen.core import Composition, periodic_table
    import numpy as np
    import pandas as pd
    import pickle

    logger.info("Generating valid compositions...") 

    #create Path to the raw data and interim data

    with open(dict_tol_factors_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    atomic_features_path: Path = RAW_DATA_DIR / "atomic_features.csv"
    electronegativities_path: Path = RAW_DATA_DIR / "electronegativities.csv"
    radii_path: Path = RAW_DATA_DIR / "Shannon_Effective_Ionic_Radii.csv"
    new_radii_path: Path = RAW_DATA_DIR / "Expanded_Shannon_Effective_Ionic_Radii.csv"
    
    df = pd.DataFrame()

    for i, cation1 in enumerate(element_symbols):
        for cation2 in element_symbols[i + 1 :]:  # Ensure unique pairs
            if cation1 != cation2:  # Ensure that two cations are not the same
                for anion in anions:
                    formula = f"{cation1}{cation2}{anion}3"  # We state here the stoichiometry of the perovskite
                    try:
                        comp = Composition(formula)
                        # Guess oxidation states
                        oxi_states_override = {anion: [-2]}
                        oxi_state_guesses = comp.oxi_state_guesses(
                            oxi_states_override=oxi_states_override
                        )

                        for guess in oxi_state_guesses:
                            if all(val is not None for val in guess.values()):
                                df.loc[comp.reduced_formula, 'A'] = cation1
                                df.loc[comp.reduced_formula, 'B'] = cation2
                                df.loc[comp.reduced_formula, 'X'] = anion
                                df.loc[comp.reduced_formula, 'nA'] = guess[cation1]
                                df.loc[comp.reduced_formula, 'nB'] = guess[cation2]
                                df.loc[comp.reduced_formula, 'nX'] = guess[anion]
                                #valid_compositions.append((comp.reduced_formula, guess))
                                break
                    except ValueError:
                        continue  # Skip invalid combinations

    electronegativities = pd.read_csv(electronegativities_path)
    new_radii = pd.read_csv(new_radii_path)
    chi_O = electronegativities.loc[electronegativities.H == 'O', '2.2' ].values
    
    for idx in df.index:
    
        nA = df.loc[idx, 'nA']
        nB = df.loc[idx, 'nB']
        nX = df.loc[idx, 'nX']

        A = df.loc[idx, 'A']
        B = df.loc[idx, 'B']
        X = df.loc[idx, 'X']

        """if nA > 0:
            nA_ = '+' + str(round(nA))
        else:
            nA_ = str(round(nA))

        if nB > 0:
            nB_ = '+' + str(round(nB))
        else:
            nB_ = str(round(nB))

        if nX > 0:
            nX_ = '+' + str(round(nX))
        else:
            nX_ = str(round(nX))"""

        Z_A = periodic_table.Element(A).Z
        Z_B = periodic_table.Element(B).Z
        Z_X = periodic_table.Element(X).Z

        #print(A + ' ' + nA_, radii.loc[(radii.ION  == A + ' ' + nA_ ) & (radii['Coord. #'] == 12), 'Ionic Radius'])
        df.loc[idx, 'chi_A'] = electronegativities.loc[electronegativities.H == A, '2.2' ].values
        df.loc[idx, 'chi_B'] = electronegativities.loc[electronegativities.H == B, '2.2' ].values
        df.loc[idx, 'chi_X'] = electronegativities.loc[electronegativities.H == X, '2.2' ].values

        #rA_ = radii.loc[(radii.ION  == A + ' ' + nA_ ) & (radii['Coord. #'] == 12), 'Ionic Radius'].values
        rA_ = new_radii.loc[(new_radii["Atomic Number"] == Z_A) & (new_radii["Oxidation State"] == nA) &(new_radii['Coordination Number'] == 12), 'Mean'].values
        
        #if np.shape(rA_)[0] == 0:
            #rA_ = radii.loc[(radii.ION  == A + nA_ ) & (radii['Coord. #'] == 12), 'Ionic Radius'].values

        if np.shape(rA_)[0] == 0:
            #rA_ = radii.loc[(radii.ION  == A + nA_ ) & (radii['Coord. #'] == 8), 'Ionic Radius'].values
            rA_ = new_radii.loc[(new_radii["Atomic Number"] == Z_A) & (new_radii["Oxidation State"] == nA) & (new_radii['Coordination Number'] == 8), 'Mean'].values

        if np.shape(rA_)[0] > 0:  
            df.loc[idx, 'rA'] = rA_

        #rB_ = radii.loc[(radii.ION  == B + ' ' + nB_ ) & (radii['Coord. #'] == 6), 'Ionic Radius'].values
        rB_ = new_radii.loc[(new_radii["Atomic Number"] == Z_B) & (new_radii["Oxidation State"] == nB) & (new_radii['Coordination Number'] == 6), 'Mean'].values

        #if np.shape(rB_)[0] == 0:
        #    rB_ = radii.loc[(radii.ION  == B + nB_ ) & (radii['Coord. #'] == 6), 'Ionic Radius'].values

        if np.shape(rB_)[0] > 0:  
            try:
                df.loc[idx, 'rB'] = rB_

            except:
                df.loc[idx, 'rB'] = rB_[0]

        #rX_ = radii.loc[radii.ION  == X + nX_, 'Ionic Radius'].values
        rX_ = new_radii.loc[(new_radii["Atomic Number"] == Z_X) & (new_radii["Oxidation State"] == nX) & (new_radii['Coordination Number'] == 6), 'Mean'].values
        df.loc[idx, 'rX'] = rX_

    df['delta_chi_AX'] = df['chi_A'] - df['chi_X']
    df['delta_chi_BX'] = df['chi_B'] - df['chi_X']

    df['delta_chi_AO'] = df['chi_A'] - chi_O
    df['delta_chi_BO'] = df['chi_B'] - chi_O

    df.dropna(inplace=True)

    df['rA_rB_ratio'] = df['rA'] / df['rB']
    df['rB_rX_ratio'] = df['rB'] / df['rX']


    df['chi_AX_ratio'] = df['delta_chi_AX'] / df['delta_chi_AO']
    df['chi_BX_ratio'] = df['delta_chi_BX'] / df['delta_chi_BO']

    df['log_rA_rB_ratio'] = np.log(df['rA_rB_ratio'])

    for tf in tolerance_factor_dict.keys():
        df.eval(tf + " = " + tolerance_factor_dict[tf][0], inplace = True)
    
    df.drop(df[df.nB < df.nA].index, inplace=True)
    
    df.to_csv(output_path)

    logger.success("%d valid compositions generated." % len(df))
    
    return df


if __name__ == "__main__":
    app()
