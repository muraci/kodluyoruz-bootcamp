###explorer:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#-----------------------------------------

def describe(df):

    dfinfo = pd.DataFrame(columns=['dtype','count','mean','std','min','median','max','null','zero','unique','freq','top','upout','lowout'])

    for col in df.columns:
        if df[col].dtypes != object:
            dfinfo.loc[col,'dtype'] = df[col].dtypes
            dfinfo.loc[col,'count'] =  ('%.0f' % df[col].count())
            dfinfo.loc[col,'mean'] = df[col].mean()
            dfinfo.loc[col,'std'] = df[col].std()
            dfinfo.loc[col,'min'] = df[col].min()
            dfinfo.loc[col,'median'] = df[col].median()
            dfinfo.loc[col,'max'] = df[col].max()
            dfinfo.loc[col,'null'] = ('%.0f' % df[col].isnull().sum())
            dfinfo.loc[col,'zero'] = ('{:.3%}'.format(df[df[col] == 0][col].count() / len(df[col])))
            dfinfo.loc[col,'unique'] = ('%.0f' % df[col].nunique())
            dfinfo.loc[col,'freq'] = ('%.0f' % df[col].value_counts().sort_values(ascending=False).values[0])
            dfinfo.loc[col,'top'] = df[col].value_counts().sort_values(ascending=False).index[0]

            Q1, Q3 = df[col].quantile(q=0.25), df[col].quantile(q=0.75)
            lowOutliers, upOutliers = Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1)
            dfinfo.loc[col, 'upout'] = ('{:.3%}'.format(df[df[col]>upOutliers][col].count()/len(df)))
            dfinfo.loc[col, 'lowout'] = ('{:.3%}'.format(df[df[col]<lowOutliers][col].count()/len(df)))
        
        elif df[col].dtypes == object:
            dfinfo.loc[col,'dtype'] = df[col].dtypes
            dfinfo.loc[col,'count'] =  ('%.0f' % df[col].count())
            dfinfo.loc[col,'null'] = df[col].isnull().sum()
            dfinfo.loc[col,'zero'] = ('{:.3%}'.format(df[df[col] == 0][col].count() / len(df[col])))
            dfinfo.loc[col,'unique'] = df[col].nunique()
            dfinfo.loc[col,'freq'] = df[col].value_counts().sort_values(ascending=False).values[0]
            dfinfo.loc[col,'top'] = df[col].value_counts().sort_values(ascending=False).index[0]

    return dfinfo


#-----------------------------------------

def unique(df, dtype = 'all', value = 100):

    ## dtype = 'all', 'obj'
    ## value = 100

    if dtype == 'all': 
        for col in df.columns:
            if (len(df[col].unique()) <= value):
                print(f'{col} : {df[col].unique()}')
                print(80*'-')
    
    elif dtype == 'obj':
        for col in df.columns:
            if (df[col].dtype == object) and (len(df[col].unique()) <= value):
                print(f'{col} : {df[col].unique()}')
                print(80*'-')
    return 


#-----------------------------------------

from scipy.stats.mstats import winsorize

def winsorized(df, col):
    
    ##upout or lowout winsorize

    Q1,Q3=df[col].quantile(q=0.25),df[col].quantile(q=0.75)
    lowOutliers,upperOutliers=Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)
    upout = (df[df[col]>upperOutliers][col].count()*100/len(df))
    lowout = (df[df[col]<lowOutliers][col].count()*100/len(df))
    if (upout != 0) | (lowout != 0):
        df[col] = winsorize(df[col], ((lowout / 100) + 1e-5, (upout / 100) + 1e-5)) #alt-üst
    
    return df


#-----------------------------------------

def quantile(df, col):

    ##25,50,75

    q25, q50, q75 = df[col].quantile(0.25), df[col].quantile(0.50), df[col].quantile(0.75)
    df.loc[(df[col] <= q25), str('q_')+col] = 0
    df.loc[(df[col] > q25) & (df[col] <= q50), str('q_')+col] = 1
    df.loc[(df[col] > q50) & (df[col] <= q75), str('q_')+col] = 2
    df.loc[(df[col] > q75), str('q_')+col] = 3
    
    return df


#-----------------------------------------

def splitby(df, col, by, unknown):

    ##directly proportional / for col 2 unique

    crossTab = pd.crosstab(df[by], df[col])
    byValue = df[by].unique()
    for val in byValue:
        idx = df.loc[(df[col] == unknown) & (df[by] == val)].index
        mask = np.random.rand(len(idx)) < ((crossTab.loc[by][df[col].unique()[0]]) / (crossTab.loc[by][df[col].unique()[1]] + crossTab.loc[by][df[col].unique()[1]]))
        idx_no, idx_yes = idx[mask], idx[~mask]
        df.loc[idx_no, col], df.loc[idx_yes, col] = df[col].unique()[0], df[col].unique()[1]
    
    return df


#-----------------------------------------

###visualizer:

def nullbar(df):

    ## null value percent
    
    for col in df.columns:
        plt.barh(col, len(df[df[col].notna()])/len(df), color = '#348ABD')
        plt.barh(col, len(df[df[col].isna()])/len(df), left = len(df[df[col].notna()])/len(df), color = '#E24A33')
    
    return plt.show()


#-----------------------------------------

def outlierbox(df):
    
    ##minmax > boxplot
    
    dfNorm = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    sns.boxplot(data = dfNorm, palette='Set3', width = 0.6)
    
    return plt.show()


#-----------------------------------------

def textbar(df):

    ##bar height annotate
    
    ax = df.plot.bar(rot=0, figsize=(15, 4), width=0.9)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for i, p in enumerate(ax.patches):
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')

    return plt.show()

#-----------------------------------------

#def multiscatter(df):
##rows and columns

#def stackbar(df):
##stacked bar plot

#def mapplot(df):
##folium map

#def stdbar(col):
##col based std and bar plot



###learner

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score, log_loss, mean_absolute_error
import statsmodels.api as sm

#-----------------------------------------

def perfstats(col, Y):

    ##rsq,mae,mse,rmse,mape

    pf = pd.DataFrame(columns=['model', 'rsq', 'rsq_adj', 'f_value', 'aic', 'bic', 'mae', 'mse', 'rmse', 'mape'])
    pd.options.display.float_format = '{:.3f}'.format
    for num,X in enumerate(col,1): 
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        
        standardscaler = StandardScaler()
        x_train = standardscaler.fit_transform(x_train)
        x_test = standardscaler.transform(x_test)
        
        x_train = sm.add_constant(x_train)
        results = sm.OLS(y_train, x_train).fit()
        x_test = sm.add_constant(x_test)
        y_pred = results.predict(x_test)
        pf.loc[num] = ('model_'+str(num) , results.rsquared, results.rsquared_adj, results.fvalue, results.aic, results.bic, 
                       mean_absolute_error(y_test, y_pred), mse(y_test, y_pred), rmse(y_test, y_pred), (np.mean(np.abs((y_test - y_pred) / y_test)) * 100))
    return pf


#-----------------------------------------

def predplts(col, Y):

    ##multi pred plot

    if(len(col) % 3) == 0:
        row = int(len(col) / 3)
    elif (len(col) % 3) != 0:
        row = int((len(col) // 3) +1)
    for num,X in enumerate(col,1): 
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        
        standardscaler = StandardScaler()
        x_train = standardscaler.fit_transform(x_train)
        x_test = standardscaler.transform(x_test)
        
        results = LinearRegression().fit(x_train, y_train)
        y_pred = results.predict(x_test)
        
        plt.subplot(row, 3, num)
        sns.scatterplot(x=y_test, y=y_pred)
        sns.lineplot(x=y_test, y=y_test, label='ytest')
        plt.ylabel("predict")
        plt.title('model_'+str(num))
        plt.tight_layout()
    
    return 


#-----------------------------------------

def regplot(X, Y, mod, idx):

    ##multi pred plot

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = False)
        
    results = mod.fit(x_train, y_train)
    y_pred = results.predict(x_test)
        
    sns.scatterplot(x=y_test, y=y_pred)
    sns.lineplot(x=y_test, y=y_test, label='ytest')
    plt.ylabel('predict')
    plt.title(str(idx))
    
    return 

#-----------------------------------------


def regframe(X, Y, mod, idx):
    
    ##rsq,mae,mse,rmse,mape
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = False)
    
    model = mod.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    k_fold = KFold(n_splits = 10, shuffle = False)

    df = pd.Series({'rsq_train': model.score(x_train, y_train),
                    'rsq_test': model.score(x_test, y_test),
                    'subt_rsq' : model.score(x_train, y_train) - model.score(x_test, y_test),
                    'mae_test': mean_absolute_error(y_test, y_pred),
                    'mse_test': mse(y_test, y_pred),
                    'rmse_test': rmse(y_test, y_pred),
                    'mape_test': (np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
                    'cross-score': cross_val_score(estimator = mod, X=X, y=Y, cv=k_fold).mean(),
                    'cross-train': cross_val_score(estimator = mod, X=x_train, y=y_train, cv=k_fold).mean()}, name = idx)
    return df

    
#-----------------------------------------

def regstats(col, Y):  #linear

    ##rsq,mae,mse,rmse,mape

    pf = pd.DataFrame(columns=['model', 'rsq_train', 'rsq_test', 'subt_rsq', 'mae_test', 'mse_test', 'rmse_test', 'mape_test']) 
       
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    
    standardscaler = StandardScaler()
    x_train = standardscaler.fit_transform(x_train)
    x_test = standardscaler.transform(x_test)


    results = LinearRegression().fit(x_train, y_train)
    y_pred = results.predict(x_test)

    pf.loc[num] = ('model_'+str(num) ,
                   results.score(x_train, y_train),
                   results.score(x_test, y_test),
                   results.score(x_train, y_train) - results.score(x_test, y_test),
                   mean_absolute_error(y_test, y_pred), 
                   mse(y_test, y_pred), 
                   rmse(y_test, y_pred), 
                   (np.mean(np.abs((y_test - y_pred) / y_test)) * 100))
    return pf


#-----------------------------------------

def coefplts(X2,Y2):

    ##coefficient bar plot

    x_train, x_test, y_train, y_test = train_test_split(X2, Y2, test_size = 0.2, random_state = 42)
    
    standardscaler = StandardScaler()
    x_train = standardscaler.fit_transform(x_train)
    x_test = standardscaler.transform(x_test)
    
    model1 = LinearRegression().fit(x_train, y_train)
    dfCoef = pd.DataFrame([model1.coef_[0:]], columns=X2.columns, index=['Linear']).T
    
    ax = dfCoef.plot.bar(rot=0, figsize=(15, 4), width=0.9)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for i, p in enumerate(ax.patches):
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')
    plt.ylim(-4, 14)
    plt.tight_layout()
    plt.show()
    
    return
    
    
#-----------------------------------------

def confusion(X, Y, mod, mix=True):
   
    ##confusion matrix, precision, recall, fscore
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = mix, random_state = 42)

    model = mod.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]
    
    dfMatrix = pd.concat([pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['pred_0', 'pred_1']), 
               pd.DataFrame(precision_recall_fscore_support(y_test, y_pred), index=['precision', 'recall', 'f1-score', 'support']).T],
               ignore_index=False, axis=1)
    print('Accuracy:', '%.5f' % model.score(x_test, y_test),'|', 'AUC:', '%.5f' % roc_auc_score(y_test, y_prob)  )
    return dfMatrix



#-----------------------------------------

#def parameter(param):
##best params


#-----------------------------------------

def modelframe(X, Y, mod, idx, mix=True):
    
    ##accuracy, precision, recall, f1-score, auc
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = mix, random_state = 42)
    
    model = mod.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    y_prob = model.predict_proba(x_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    k_fold = KFold(n_splits = 10, shuffle = mix, random_state = 42)

    df = pd.Series({'train-score': accuracy_score(y_train,y_train_pred),
                    'test-score': accuracy_score(y_test,y_pred),
                    'precision' :precision_score(y_test,y_pred),
                    'recall': recall_score(y_test,y_pred),
                    'f1-score': f1_score(y_test,y_pred),
                    'auc-roc': roc_auc_score(y_test, y_prob),
                    'auc-pr': auc(recall, precision),
                    'cross-score': cross_val_score(estimator = model, X=X, y=Y, cv=k_fold).mean(),
                    'cross-train': cross_val_score(estimator = model, X=x_train, y=y_train, cv=k_fold).mean()}, name = idx)
    return df


#-----------------------------------------

def modelcurve(X, Y, mod, idx, mix=True) :
         
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = mix, random_state = 42)

    model = mod.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]

    plt.subplot(1, 2, 1)
    fpr, tpr, thresholds  = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label = idx + ' %.3f' % roc_auc_score(y_test, y_prob))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)  
    plt.plot(precision, recall, label = idx + ' %.3f' % auc(recall, precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall/Precision Curve')
    plt.legend()
    
    return
