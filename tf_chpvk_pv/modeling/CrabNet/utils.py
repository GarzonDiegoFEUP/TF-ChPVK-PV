from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np

from tf_chpvk_pv.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, TRAINED_MODELS, CRYSTALLM_DATA_DIR

app = typer.Typer()
@app.command()
def main():
    logger.info("Loading raw data...")
    raw_data_path = RAW_DATA_DIR / "chpvk_pv_data.csv"
    df = pd.read_csv(raw_data_path)

    logger.info("Processing data...")
    # Example processing: Fill missing values and normalize bandgap
    df['bandgap'] = df['bandgap'].fillna(df['bandgap'].mean())
    df['bandgap_normalized'] = (df['bandgap'] - df['bandgap'].min()) / (df['bandgap'].max() - df['bandgap'].min())

    logger.info("Saving processed data...")
    processed_data_path = PROCESSED_DATA_DIR / "chpvk_pv_data_processed.csv"
    df.to_csv(processed_data_path, index=False)

    logger.info("Data processing complete.")


def get_raw_data(input_path_pvk: Path = RAW_DATA_DIR / "perovskite_bandgap_devices.csv",
             input_path_chalcogenides: Path = RAW_DATA_DIR / "chalcogenides_bandgap_devices.csv",
             input_path_chalc_semicon: Path = RAW_DATA_DIR / "chalcogen_semicon_bandgap.csv",
             output_path: Path = PROCESSED_DATA_DIR / "chpvk_dataset.csv",
             new_radii_path: Path = RAW_DATA_DIR / "Expanded_Shannon_Effective_Ionic_Radii.csv",
             turnley_radii_path: Path = RAW_DATA_DIR / "Turnley_Ionic_Radii.xlsx",):

    
    df_pvk = pd.read_csv(input_path_pvk)
    df_chalcogenides = pd.read_csv(input_path_chalcogenides)
    df_chalc_semicon = pd.read_csv(input_path_chalc_semicon)
    for df_, sc in zip([df_pvk, df_chalcogenides, df_chalc_semicon], ['pvk', 'chalcogenides', 'chalc_semicon']):
        df_['source'] = sc
    df = pd.concat([df_pvk, df_chalcogenides, df_chalc_semicon])
    sources = (x*100/df.shape[0] for x in [df_pvk.shape[0], df_chalcogenides.shape[0], df_chalc_semicon.shape[0]])
    txt = 'The data comes from the following sources:\n{0:.2f} % from halide perovskites,\n{1:.4f} % from chalcogenides perovskites,\n{2:.2f} % from chalcogenide semiconductors'.format(*sources)
    print(txt)

    df = df[df['bandgap'].notna()]
    df = df[df['reduced_formulas'].notna()]

    return df

def save_processed_data(df: pd.DataFrame,
                        output_path: Path = INTERIM_DATA_DIR / 'df_grouped_formula_complete_dataset.csv',):
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def get_petiffor_features(df_grouped_formula,
                          input_petiffor_path: Path = RAW_DATA_DIR / 'petiffor_embedding.csv',
                          train=True,
                          original_df: pd.DataFrame = None,):
   
  # Add petifor embedding
  from pymatgen.core import Composition
  from ase.atom import Atom
  from sklearn.model_selection import train_test_split


  petiffor = pd.read_csv(input_petiffor_path, index_col=0)

  def get_onehot_comp(composition, elemental_embeddings):
    if isinstance(composition, str):
      composition = Composition(composition)
    a = composition.fractional_composition.get_el_amt_dict()
    comp_finger =  np.array([a.get(Atom(i).symbol, 0) for i in range(1,99)])
    comp_finger = comp_finger @ elemental_embeddings.values
    return comp_finger

  df_grouped_formula['petiffor'] = df_grouped_formula.formula.apply(lambda x: get_onehot_comp(x, petiffor))

  df = df_grouped_formula.copy()
  size_pf = df.petiffor[0].shape[0]
  feature_names = ['petiffor_' + str(i) for i in range(0, size_pf)]
  new_df = pd.DataFrame(columns = feature_names, index=df.index)

  for idx, arr in enumerate(df.petiffor.values):
      new_df.iloc[idx] = arr

  df = pd.concat([df, new_df.astype('float64')], axis=1)
  df.drop(columns=['petiffor'], inplace=True)

  if train:

    #add_source to do the separation
    if 'source' not in df.columns and original_df is not None:
     for formula in df['formula']:
        df.loc[df['formula'] == formula, 'source'] = original_df.loc[original_df['formula'] == formula, 'source'].values[0]

    try:
      # First split: 80% train, 20% temp (val+test), stratified by source
      train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['source'], random_state=42
      )
      # Second split: split temp into 50/50 -> 10% val, 10% test, stratified by source
      val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['source'], random_state=42
      )
    except ValueError:
      # Fallback to non-stratified shuffle split if stratification is not possible
      train_df, val_df, test_df = np.split(
        df.sample(frac=1, random_state=42),
        [
            int(0.8 * len(df)),
            int(0.9 * len(df))
        ]
      )
    
    if 'source' in train_df.columns:
      train_df.drop(columns=['source'], inplace=True)
      val_df.drop(columns=['source'], inplace=True)
      test_df.drop(columns=['source'], inplace=True)

    return train_df, val_df, test_df, feature_names
  else:
    return df
  

def load_model(model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth',):
    from crabnet.crabnet_ import CrabNet  # type: ignore
    from crabnet.kingcrab import SubCrab  # type: ignore

    # Instantiate SubCrab
    sub_crab_model = SubCrab()

    # Instantiate CrabNet and set its model to SubCrab
    crabnet_model = CrabNet()
    crabnet_model.model = sub_crab_model

    # Load the pre-trained network
    crabnet_model.load_network(str(model_path))
    crabnet_model.to('cuda')

    return crabnet_model

def get_test_r2_score_by_source_data(df, original_df,
                                     feature_names,
                                     crabnet_bandgap = None,
                                     model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth',):

    for formula in df['formula']:
        df.loc[df['formula'] == formula, 'source'] = original_df.loc[original_df['formula'] == formula, 'source'].values[0]

    sources = original_df['source'].unique().tolist()
    for source in sources:
        df_source = df[df['source'] == source]
        print(f'\nResults for source: {source} with data size {df_source.shape[0]}')
        if df_source.shape[0] > 0:
            test_r2_score(df_source,
                          feature_names,
                          crabnet_bandgap=crabnet_bandgap,
                          model_path=model_path)
        else:
            print('No data available for this source.')

def test_r2_score(df,
                  feature_names,
                  crabnet_bandgap = None,
                  model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth',):
  
  from crabnet.utils.figures import act_pred  # type: ignore
  import pandas as pd
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  


  if not crabnet_bandgap:
    crabnet_bandgap = load_model(model_path=model_path)

  # Train data
  df_zeros = pd.DataFrame({"formula": df['formula'], "target": [0.0]*len(df['formula']),})
  df_zeros = pd.concat([df_zeros, df[feature_names]], axis=1)

  df_predicted, df_predicted_sigma = crabnet_bandgap.predict(df_zeros, return_uncertainty=True)

  act_pred(df['target'], df_predicted)
  r2 = r2_score(df['target'], df_predicted)
  print(f'R2 score: {r2}')
  mse = mean_squared_error(df['target'], df_predicted)
  print(f'MSE: {mse}')
  mae = mean_absolute_error(df['target'], df_predicted)
  print(f'MAE: {mae} eV')

def predict_bandgap(formula, 
                     crabnet_model = None,
                     model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth',):
    
  if not crabnet_model:
    crabnet_model = load_model(model_path=model_path)

  input_df = pd.DataFrame({"formula": [formula], "target": [0.0]})
  input_df = get_petiffor_features(input_df, train=False)
  prediction, prediction_sigma = crabnet_model.predict(input_df, return_uncertainty=True)
  return prediction, prediction_sigma

def get_CrystaLLM_predictions(crabnet_model = None,
                              input_data_CrystaLLM: Path = CRYSTALLM_DATA_DIR / 'results CrystaLLM.csv',
                              output_data_CrystaLLM: Path = PROCESSED_DATA_DIR / 'results_CrystaLLM_with_bandgap.csv',):

  if not crabnet_model:
    crabnet_model = load_model()

  df_compositions = pd.read_csv(input_data_CrystaLLM)
  df_compositions.rename(columns={'material': 'formula'}, inplace=True)
  df_compositions.set_index('formula', inplace=True)
  for formula in df_compositions.index:
      prediction, prediction_sigma = predict_bandgap(formula, crabnet_model)
      df_compositions.loc[formula, 'bandgap'] = prediction
      df_compositions.loc[formula, 'bandgap_sigma'] = prediction_sigma

  df_compositions.to_csv(output_data_CrystaLLM)
  
  return df_compositions

def get_SISSO_predictions(crabnet_model = None,
                          input_data_SISSO: Path = PROCESSED_DATA_DIR / 'stable_compositions.csv',
                          output_data_SISSO: Path = PROCESSED_DATA_DIR / 'results_SISSO_with_bandgap.csv'):

  if not crabnet_model:
    crabnet_model = load_model()

  df_compositions = pd.read_csv(input_data_SISSO)
  df_compositions.rename(columns={'Unnamed: 0': 'formula'}, inplace=True)
  df_compositions.set_index('formula', inplace=True)
  for formula in df_compositions.index:
      prediction, prediction_sigma = predict_bandgap(formula, crabnet_model)
      df_compositions.loc[formula, 'bandgap'] = prediction
      df_compositions.loc[formula, 'bandgap_sigma'] = prediction_sigma

  df_compositions.to_csv(output_data_SISSO)
  
  return df_compositions

def get_experimental_predictions(crabnet_model = None,
                                 input_data_experimental: Path = RAW_DATA_DIR / 'chalcogenides_bandgap_devices.csv',
                                 output_data_experimental: Path = PROCESSED_DATA_DIR / 'results_experimental_with_bandgap.csv',):

  if not crabnet_model:
    crabnet_model = load_model()

  df_chalcogenides = pd.read_csv(input_data_experimental)
  for formula in df_chalcogenides.descriptive_formulas.unique():
      prediction, prediction_sigma = predict_bandgap(formula)
      print(f'Experimental bandgap for {formula}:', str(df_chalcogenides.loc[df_chalcogenides['descriptive_formulas'] == formula, 'bandgap'].values[0]) + ' eV')
      print(f'Bandgap prediction for {formula}:', f"{round(prediction[0], 2)} ± {round(prediction_sigma[0], 2)}" + ' eV')

      df_chalcogenides.loc[df_chalcogenides['descriptive_formulas'] == formula, 'predicted_bandgap'] = prediction[0]
      df_chalcogenides.loc[df_chalcogenides['descriptive_formulas'] == formula, 'predicted_bandgap_sigma'] = prediction_sigma[0]

  df_chalcogenides.to_csv(output_data_experimental)
    
  return df_chalcogenides



if __name__ == "__main__":
    app()
