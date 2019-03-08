import numpy as np


def compute_means_and_stds( dataframes, columns=None ):
  '''
  Takes in a list of dataframes
  Assume that all dataframes has the same column names and same amount of columns
  Then for the parameter columns (list of column names or indexes)
    iteraterate over each dataframe in the dataframes-list, and get a copy of that iteratations dataframe columns that where specified in the columns parameter,
    add the columns copy into the new list, containing all the copies of columns from all the dataframes in the list of passed in dataframes.
    # np.concatenate --> a = [[1,2], [3,4]] b = [[5,6]], c = np.concatatenate((a,b), axis=0 #row) --> c = [[1,2],[3,4],[5,6]]

  Then it adds to the mean list  the calculated mean (arithmetic mean (basic average)) over all the lists it just combined, thus retrurning just ONE number!
  Then  it adds to the stds list the calculated std (Standard deviation, aka spread of a distribution of the array elements) of the combined list of columns, thus returning just ONE number!

  When it has done this for all the columns specificed in the parameter columns, over all the dataframes in the parameter dataframes
  it returns the lists, which are 2d
  means = [a, b]
  stds = [x, y]

  :param dataframes: [pd.dataframe1, pd.dataframe2, ... pd.dataframeN]
  :param columns: ['c1', 'c2', 'c3']
  :return: [a,b], [x,y]
  '''

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