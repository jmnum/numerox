import pandas as pd
import numpy as np

def spearmanr(target, pred):
    return np.corrcoef(target, pred.rank(pct=True, method="first") )[0, 1]
    
 def ar1(x):
    return np.corrcoef(x.iloc[:-1], x.iloc[1:])[0,1]
    
def score_correlation(labels, predictions):
    if type(predictions) is not pd.core.frame.DataFrame:
        predictions_df = pd.DataFrame(predictions)
    else:
        predictions_df = predictions
    ranked_predictions = predictions_df.rank(pct=True, method='first')

    return np.corrcoef(labels, np.array(ranked_predictions)[:, 0])[0, 1]
    
def era_scores(data):
   new_df = data['validation'].df[['era','target']].copy()
   scores = pd.Series(index=new_df["era"].unique(),dtype='float32')
   for era in new_df["era"].unique():
      era_df = new_df[new_df["era"] == era]
      scores[era] = spearmanr( era_df["target"],era_df["pred"])
   era_scores.sort_values(inplace=True)
