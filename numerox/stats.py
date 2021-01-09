import pandas as pd
import numpy as np

def spearmanr(target, pred):
    return np.corrcoef(target, pred.rank(pct=True, method="first") )[0, 1]
    
def ar1(x):
    return np.corrcoef(x.iloc[:-1], x.iloc[1:])[0,1]
    
def correlation_per_era(data,ppname):
   new_df=ppname.df
   new_df.columns=['pred']
   new_df=new_df.join(data.region_isin(['train','validation']).df[['era','target']],how='inner')
   era_scores = pd.Series(index=new_df["era"].unique(),dtype='float32')
   for era in new_df["era"].unique():
      era_df = new_df[new_df["era"] == era]
      era_scores[era] = spearmanr( era_df["target"],era_df["pred"])
  
   return era_scores.to_frame(name='corr')

def mean_correlation(data,ppname):
    era_scores= correlation_per_era(data,ppname)['corr']
    return era_scores.mean()

def sharpe(data,ppname):
    era_scores= correlation_per_era(data,ppname)['corr']
    return era_scores.mean()/era_scores.std()
    
def autocorrelation(data,ppname):
    era_scores= correlation_per_era(data,ppname)['corr']
    return ar1(era_scores)

def FeatExp(data,ppname):
    new_df=ppname.df
    new_df.columns=['pred']
    new_df=new_df.join(data.region_isin(['train','validation']).df,how='inner')
    correlations = pd.Series(dtype='float32')
    for feature in new_df.filter(like='feature'):
        feat_corr=np.corrcoef(new_df['pred'], new_df[feature])[0,1]
        correlations[feature]=feat_corr
  
    return correlations.to_frame(name='corr')
   
