from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV 
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, TREES_DIR, RESULTS_DIR

app = typer.Typer()


@app.command()
def main():
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    t_sisso_expression = train_tree_sis_features()
    logger.success("Modeling training complete.")

    logger.info("Evaluating t_sisso...")
    train_df, test_df, tolerance_factor_dict = evaluate_t_sisso(t_sisso_expression)

    tfs = ['t_sisso', 't', 'tau', 't_jess']
    tf_tresh = [1, 2, 1, 2]

    df_acc =pd.DataFrame()
    clfs = {}

    for tf, tresh in zip(tfs, tf_tresh):
        df_acc, clf_t = test_tolerance_factor(tf, train_df, test_df, tolerance_factor_dict, df_acc, n_tresh=tresh)
        clfs[tf] = clf_t

    df_acc.to_csv(RESULTS_DIR / 'tolerance factors accuracy.csv')

    logger.success("Modeling evaluation complete.")

    train_platt_scaling(train_df, test_df, tolerance_factor_dict, clfs['t_sisso'])
    # -----------------------------------------


def train_platt_scaling(train_df, test_df, tolerance_factor_dict, clf_t,
                        output_dir: Path = RESULTS_DIR,
                        tolerance_factor_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl"):

    logger.info("Training Platt scaling model...")

    with open(tolerance_factor_path, 'wb') as file:
        pickle.dump(tolerance_factor_dict, file)

    x_train_t_sisso = train_df['t_sisso'].to_numpy()
    x_test_t_sisso = test_df['t_sisso'].to_numpy()
    threshold_t_sisso = tolerance_factor_dict['t_sisso'][1]

    labels_platt=clf_t.predict(x_train_t_sisso.reshape(-1,1))
    clf2_sisso = CalibratedClassifierCV(cv=3)
    clf2_sisso = clf2_sisso.fit(x_train_t_sisso.reshape(-1,1), labels_platt)
    p_t_sisso_train=clf2_sisso.predict_proba(x_train_t_sisso.reshape(-1,1))[:,1]
    p_t_sisso_test=clf2_sisso.predict_proba(x_test_t_sisso.reshape(-1,1))[:,1]
    train_df['p_t_sisso'] = p_t_sisso_train            # add p_t_sisso to the train and test data frame
    test_df['p_t_sisso'] = p_t_sisso_test

    train_df.to_csv(output_dir / 'processed_chpvk_train_dataset.csv')
    test_df.to_csv(output_dir / 'processed_chpvk_test_dataset.csv')

    logger.success("Platt scaling model training complete.")

    return train_df, test_df


def train_tree_sis_features(
    features_path: Path = INTERIM_DATA_DIR / "features_sisso.csv",
    train_data_path: Path = PROCESSED_DATA_DIR / "chpvk_train_dataset.csv",
    test_data_path: Path = PROCESSED_DATA_DIR / "chpvk_test_dataset.csv",
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training tree model with SISSO features...")
    # -----------------------------------------

    #train classification trees for the selected descriptors
    train_df = pd.read_csv(train_data_path, index_col=0)
    test_df = pd.read_csv(test_data_path, index_col=0)
    feature_df = pd.read_csv(features_path, index_col=0)



    #depth of the classification tree - user has to choose
    def rank_tree(labels, feature_space, depth):                       #rank features according to the classification-tree accuracy                             
        score = []
        for i in list(range(0,feature_space.shape[1])):                # 'i' is a column and 'for' is from the first column to the last one
            x=np.array(feature_space)[:,i]                             # take the first column values
            clf = tree.DecisionTreeClassifier(max_depth=depth, class_weight='balanced', criterion='entropy')
            
            clf_cv = cross_validate(clf, x.reshape(-1,1), labels, scoring='accuracy')
            clf_cv_score = np.mean(clf_cv['test_score'])
            
            #clf = clf.fit(x.reshape(-1,1), labels)                     # Build a decision-tree classifier from the training set (X, y). X is the values of features (for each for iteration on column) and Y is the target value, here exp_label
            score.append([feature_space.columns.values[i],clf_cv_score])      # make a list of the feature and the mean accuracy of the all values of that feaure (for different materials)
        score_sorted=sorted(score,reverse=True,key=lambda x: x[1])     # sort the features based on the accuracy
        return score_sorted

    labels_train=train_df["exp_label"].to_numpy()
    labels_test=test_df["exp_label"].to_numpy()
    rank_list=rank_tree(labels_train, feature_df, 1)
    tree_pd =pd.DataFrame(rank_list, columns=['feature','tree accuracy'])       # make a new data frame of the feature and the accuracies

    # the first ranked feature is the t_sisso
    t_sisso_expression=str(rank_list[0][0])   
    t_sisso_expression = t_sisso_expression.replace("ln","log")
    t_sisso_expression = t_sisso_expression.replace("^","**")
    print('Identified expression for t_sisso: %s' % t_sisso_expression)
    return t_sisso_expression
    
def evaluate_t_sisso(t_sisso_expression,
                     train_df_path: Path = PROCESSED_DATA_DIR / "chpvk_train_dataset.csv",
                     test_df_path: Path = PROCESSED_DATA_DIR / "chpvk_test_dataset.csv"):

    train_df = pd.read_csv(train_df_path, index_col=0)
    test_df = pd.read_csv(test_df_path, index_col=0)

    
    #make a dictionary for t_sisso,t, tau
    tolerance_factor_dict = {
    "t_sisso": [t_sisso_expression],
    "t": ["(rA+rX)/(1.41421*(rB+rX))"],
    "tau": ["rX/rB-nA*(nA-rA_rB_ratio/log(rA_rB_ratio))"],
    "t_jess": ["chi_AX_ratio * (rA+rX)/(1.41421*chi_BX_ratio*(rB+rX))"],
    #"t_old": ["sqrt(chi_AX_ratio) * 1/log(rA_rB_ratio) - (rB_rX_ratio * nB) + (rA_rB_ratio/chi_AX_ratio)"],
    }


    #Add tau threshold
    tolerance_factor_dict["tau"].append([4.18])
    #tolerance_factor_dict["t_old"].append(2.75)

    train_df.eval('t_sisso = ' + tolerance_factor_dict['t_sisso'][0], inplace=True)
    train_df.eval('t = '+ tolerance_factor_dict['t'][0],inplace=True)
    train_df.eval('tau = '+ tolerance_factor_dict['tau'][0], inplace=True) 
    train_df.eval('t_jess = '+ tolerance_factor_dict['t_jess'][0], inplace=True) 
    #train_df.eval('t_old = '+ tolerance_factor_dict['t_old'][0], inplace=True) 

    test_df.eval('t_sisso = ' + tolerance_factor_dict['t_sisso'][0], inplace=True)
    test_df.eval('t = '+ tolerance_factor_dict['t'][0], inplace=True)
    test_df.eval('tau = '+ tolerance_factor_dict['tau'][0], inplace=True)
    test_df.eval('t_jess = '+ tolerance_factor_dict['t_jess'][0], inplace=True)
    #test_df.eval('t_old = '+ tolerance_factor_dict['t_old'][0], inplace=True)



    return train_df, test_df, tolerance_factor_dict

def test_tolerance_factor(t, train_df, test_df, tolerance_factor_dict, df_acc=pd.DataFrame(),
                          n_tresh=1, cl_w = 'balanced', crit='entropy'):
    
    labels_train=train_df["exp_label"].to_numpy()
    labels_test=test_df["exp_label"].to_numpy()
    
    if len(df_acc.columns) == 0:
        df_acc = pd.DataFrame(columns=['t_sisso', 't', 'tau', 't_jess'], index=['train_data', 'test_data', 'all_data'] )
        
    x_train_=train_df[t].to_numpy()
    x_test_=test_df[t].to_numpy()
    
    
    clf1_tree = tree.DecisionTreeClassifier(max_depth=n_tresh, class_weight=cl_w, criterion=crit)
    
    clf1_cv = cross_validate(clf1_tree, x_train_.reshape(-1,1), labels_train, scoring='accuracy')
    
    clf1_cv_score = np.mean(clf1_cv['test_score'])
    
    clf1_model = clf1_tree.fit(x_train_.reshape(-1,1),labels_train)
    labels_pred_=clf1_model.predict(x_test_.reshape(-1,1))
    tree.plot_tree(clf1_model)

    name_figure = 'tree_' + t + '.png'
    
    plt.savefig(TREES_DIR / name_figure)
    
    acc_train = clf1_model.score(x_train_.reshape(-1,1),labels_train)
    acc_test = metrics.accuracy_score(labels_test, labels_pred_)
    
    #General accuracy
    all_data = np.append(x_train_, x_test_)
    all_labels = np.append(labels_train, labels_test)
    labels_all_data = clf1_model.predict(all_data.reshape(-1,1))
    acc_all = metrics.accuracy_score(all_labels, labels_all_data)
    
    print('Classification tree accuracy (for ' + t + ') on the train set: %f.' % acc_train)
    print('Classification tree accuracy (for ' + t + ') on the train set (5 fold CV): %f.' % clf1_cv_score)
    print('Classification tree accuracy (for ' + t + ') on the test set: %f.' % acc_test)

    if n_tresh == 2:
        threshold_=[clf1_model.tree_.threshold[0],clf1_model.tree_.threshold[4]]

        # add t threshold to dictionary
        tolerance_factor_dict[t].append(threshold_)

        print('%f < ' % tolerance_factor_dict[t][1][0] +  t +' < %f indicates stable perovskites.' % tolerance_factor_dict[t][1][1])
    elif n_tresh == 1:
        threshold_= clf1_model.tree_.threshold[0]

        #Add threshold to the dictionary
        tolerance_factor_dict[t].append(threshold_)

        print(t + ' < %f indicates stable perovskites.' % tolerance_factor_dict[t][1])
              
    df_acc.loc['train_data', t] = acc_train
    df_acc.loc['test_data', t] = acc_test
    df_acc.loc['all_data', t] = acc_all
    
    #get accuracy per X anion
    dict_ch = {1.33:'F',
               1.81:'Cl',
               1.98:'Se',
               1.96:'Br',
               1.84:'S',
               2.2:'I'
              }       
    
    for rx in train_df.rX.unique():
        x_test_ch = test_df.loc[test_df.rX == rx, t].to_numpy()
        x_train_ch = train_df.loc[train_df.rX == rx, t].to_numpy()
        
        labels_test_ch = labels_test[test_df.rX == rx]
        labels_train_ch = labels_train[train_df.rX == rx]
        
        labels_pred_ch=clf1_model.predict(x_test_ch.reshape(-1,1))
        
        acc_train_ch = clf1_model.score(x_train_ch.reshape(-1,1),labels_train_ch)
        acc_test_ch = metrics.accuracy_score(labels_test_ch, labels_pred_ch)
        
        df_acc.loc['train_data_' + dict_ch[rx], t] = acc_train_ch
        df_acc.loc['test_data_' + dict_ch[rx], t] = acc_test_ch 
        
    df_acc.loc['5-fold CV', t] = clf1_cv_score
    
    return df_acc, clf1_model



if __name__ == "__main__":
    app()
