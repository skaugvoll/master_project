import collections

import pandas as pd 
import numpy as np 

class DataEncoder:

  def __init__( self, classes ):
    # Make sure that the class configuration is valid
    self.validate_input( classes )
    # Store classes as provided
    self.classes = classes
    # A Mapping from class value to name, e.g.: { 6 -> standing }
    self.name_lookup = { cl['value']:cl['name'] for cl in self.classes }
    # A Mapping from class name to value, e.g.: { standing -> 6 }
    self.value_lookup = { cl['name']:cl['value'] for cl in self.classes }
    # A list of non-replaced classes
    self.active_classes = [ cl for cl in self.classes if not 'replace_by' in cl ] 
    # A set of valid raw labels 
    self.valid_raw_labels = [ cl['value'] for cl in self.classes ]
    # A set of valid labels after replacing
    self.valid_labels = [ cl['value'] for cl in self.active_classes ]
    # A dictionary that goes { raw_label -> one_hot_index }
    self.one_hot_enc_lookup = self.get_one_hot_encoding_dict( self.active_classes, self.classes )

  @property
  def num_active_classes( self ):
    return len( self.active_classes )

  def get_one_hot_indexes( self, targets ):
    '''
    Replace labels in targets by one-hot indexes
    E.g.: Suppose you have valid classes [1,2,3,4]
          [1,1,3,1] -> [0,0,2,0]
    '''
    return pd.Series( targets ).replace( self.one_hot_enc_lookup ).values

  def compute_class_weights( self, targets, norm=1 ):
    '''
    Utility function for computing weights for each class.
    The amount of weight for a class will be inversely proportional 
    to the frequency of that class.
    
    Inputs:
      - targets
        An array of unencoded targets, with values as they appear in
        the class definition
      - norm=1
        Optional parameter for equalizing the distribution. The 'norm-th'
        root of the class frequency distribution will be taken, allowing 
        the user a way to prevent the result from being too aggresively
        in favour of minority classes
        
    Outputs:
      An array of weight of length equal to the length of a single
      one-hot encoded datapoint, ready to be fed into Keras model.fit
    '''
    # Get one-hot indexes so that all labels are in [0,#classes-1]
    one_hot_indexes = self.get_one_hot_indexes( targets )
    # Compute counts of each one-hot index
    bincounts = np.bincount( one_hot_indexes, minlength=self.num_active_classes )
    # Add one to each count to avoid dividing by zero
    bincounts += 1
    # Check that we got the correct length
    assert len( bincounts ) == self.num_active_classes
    # Compute by taking the reciprocal of each probability
    # Also take the "norm-th" root of it to equalize the distribution
    weights = ( len(targets)/bincounts ) ** ( 1/norm )
    # Make sure that weights sum up to #classes
    weights = weights * len( bincounts ) / weights.sum()
    
    return weights
    


  def one_hot_encode( self, targets ):
    '''
    Encodes input targets as one-hot vectors
    E.g.: Suppose you have valid classes [1,2,3,4]
          [1,1,3,1] -> [[1,0,0,0],
                        [1,0,0,0],
                        [0,0,1,0],
                        [1,0,0,0]]
    '''
    one_hots = np.zeros(( len(targets), len( self.valid_labels )))
    one_hots[ range(len(targets)), self.get_one_hot_indexes( targets ) ] = 1
    return one_hots

  def one_hot_decode( self, one_hots ):
    '''

    '''
    return one_hots.argmax( axis=1 ).choose( self.valid_labels )

  def check_valid( self, targets ):
    ''' 
    Check that a list of labels only contain recognized classes
    '''
    valid = pd.Series( targets ).isin( self.valid_raw_labels )
    if not valid.all():
      raise ValueError( 'Some classes in provided targets are not in: %s'%self.valid_raw_labels )


  def get_one_hot_encoding_dict( self, active_classes, all_classes ):
    '''
    Get a dictionary that maps from each recognized class
    '''
    # Initial encoding dict for non-replaced classes
    one_hot_enc = { cl['value']:i for i,cl in enumerate( active_classes )}
    # Make a temporary lookup; { name -> value }
    name_to_value = { cl['name']:cl['value'] for cl in active_classes }
    # Add replaced classes
    for cl in all_classes:
      # Ignore if already in dict, i.e. no replaced_by field
      if cl['value'] in one_hot_enc: continue 
      # Use one hot index of replaced class
      replaced_by_value = name_to_value[ cl['replace_by']]
      one_hot_enc[ cl['value'] ] = one_hot_enc[ replaced_by_value ]

    return one_hot_enc


  def validate_input( self, classes ):

    # Check that all class names are unique
    names = [ cl['name'] for cl in classes ]
    duplicate_names = [ name for name,c in collections.Counter(names).items() if c > 1 ]
    if duplicate_names:
      raise ValueError( 'Duplicated class names provided; offenders: %s'%duplicate_names )

    # Check that all class values are unique
    values = [ cl['value'] for cl in classes ]
    duplicate_values = [ value for value,c in collections.Counter(values).items() if c > 1 ]
    if duplicate_values:
      raise ValueError( 'Duplicated class values provided; offenders: %s'%duplicate_values )

    # Check that all replace_by fields refer to an existing class 
    names  = set( cl['name'] for cl in classes )
    missing = [ cl for cl in classes if 'replace_by' in cl and cl['replace_by'] not in names ]
    if missing:
      raise ValueError( 'Some classes have non-existing "replace_by" fields; offenders: %s'%missing )

    # Check that no class has "replace_by" pointing to another class with "replace_by"
    replace_dict = { cl['name']:cl['replace_by'] for cl in classes if 'replace_by' in cl }
    double_replaces = [ name for name in replace_dict if replace_dict[name] in replace_dict ]
    if double_replaces:
      raise ValueError( 'Some classes are replaced by classes that are themselves replaced; offenders: %s'%double_replaces )
