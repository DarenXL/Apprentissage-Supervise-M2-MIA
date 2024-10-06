import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# **********************************************
#
#                VISUALISATION
#
# **********************************************

def corrplot(df, target_var):
    """
    > Plot correlation matrix on dataframe df with target_var
    """
    X = df.drop(target_var, axis=1)
    y = df[target_var]
    size = len(X.columns)

    num_cols, mt_width = 7, 2
    mt_heigh = 1 + size//(mt_width * num_cols)
    
    fig, ax = plt.subplots(mt_heigh, mt_width, figsize=(8, 4*mt_heigh))
    plt.subplots_adjust(wspace=1, hspace=1)
    fig.suptitle("Correlation matrix", size=15)

    for i in range(mt_heigh):
        for j in range(mt_width):
            k = mt_width * i + j
            size_k = min(num_cols, (size - k*num_cols))
            ax[i, j].plot([size_k, size_k], [0, size_k], color='black')
            ax[i, j].plot([0, size_k], [size_k, size_k], color='black')
            sns.heatmap(
                pd.concat( (X.iloc[:,k*num_cols:(k+1)*num_cols], y), axis=1 ).corr(), 
                annot=False, 
                cmap='coolwarm', 
                linewidths=0.25, 
                vmin=-1, 
                vmax=1, 
                ax=ax[i, j]
            )
    
def biScatterPlot(var_name, target_var, df):
    """
    > BiScatter plot for target_var depending on var_name
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Correlation {target_var} - {var_name} ", size=15)

    ax = axs[0]
    ax.grid()
    ax.scatter(df[var_name], df[target_var], marker='.', color='orange')
    ax.set_xlabel(var_name)
    ax.set_ylabel(target_var)

    ax = axs[1]
    ax.grid()
    ax.scatter(np.log(df[var_name] + 1), df[target_var], marker='.', color='green')
    ax.set_xlabel(f'log({var_name} + 1)')
    ax.set_ylabel(target_var)

def colorScatterPlot(var_name1, var_name2, target_var, df):
    """
    > Plot for var_name1 and var_name2 with gradient color depending on target_var
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle(f"Color plot on {target_var}", size=15)

    ax.grid()
    ax.scatter(df[var_name1], df[var_name2], c=df[target_var], marker='.', cmap='viridis', label=target_var)
    ax.set_xlabel(var_name1)
    ax.set_ylabel(var_name2)
    ax.legend()

# **********************************************
#
#               CROSS VALIDATION
#
# **********************************************

def GridSearchResults(model, Xtrain, Ytrain, scoring, select_vars=None, n_folds=5, param_grid={}, plot=False, refit=None):
    
    # metrique de réévaluation du meilleur modèle
    if refit == None : refit = list(scoring.keys())[0]
    
    # figure de visualisation des résultats
    if plot : fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # select vars and grid search
    if select_vars != None : grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, refit='r2', cv=5).fit(Xtrain[select_vars], Ytrain)
    else : grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, refit='r2', cv=5).fit(Xtrain, Ytrain)
    
    # resultats détaillés
    results = grid_search.cv_results_
    
    # affichage dans un dataframe
    data_res = pd.DataFrame({})
    data_res.index = results['params']
    data_res['fit time'] = [f'{results['mean_fit_time'][i]:.3f} ± {results['std_fit_time'][i]:.3f}' for i in range(len(results['mean_fit_time']))]
    for metric in scoring.keys():
        data_res[metric] = [f'{abs(results['mean_test_' + metric][i]):.3f} ± {results['std_test_' + metric][i]:.3f}' for i in range(len(results['mean_test_' + metric]))]
        if plot and metric == refit : ax.plot(range(0, len(data_res.index)), np.round(np.abs(results['mean_test_' + metric]), 3), marker='o', label=metric + '_cv_test')

    if plot : 
        ax.grid()
        ax.legend()
        ax.set_xlabel("params_index")
        ax.set_ylabel(refit)
        
    # retour des résultats
    return data_res

# **********************************************
#
#                  SUBMISSION
#
# **********************************************

def submit_model(filename, Ypred):
    pd.DataFrame({

        'row_ID':range(0, len(Ypred)),
        'tip_amount':Ypred

    }).to_parquet('predictions/' + filename + '.parquet', index=False)





