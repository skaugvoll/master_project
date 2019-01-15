import numpy as np


def compute_means_and_stds( dataframes, columns=None ):
  
  means = []
  stds  = []

  for col in ( columns or dataframes[0].columns ):
    combined = np.concatenate([ df[col] for df in dataframes ])
    means.append( np.mean( combined ))
    stds.append( np.std( combined ))

  return means, stds 



def normalize( subjects, means_and_stds ):

  for subject in subjects:
    for col, mean_and_std in means_and_stds.items():
      subject.features[col] -= mean_and_std['mean']
      subject.features[col] /= mean_and_std['std']

  return subjects