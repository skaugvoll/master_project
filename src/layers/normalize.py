from keras.layers import Layer, Input, Dense
from keras import initializers
from keras.models import Model

import numpy as np


class Normalize( Layer ):
    
  def __init__( self, shape=None, trainable=False, **kwargs ):
    super( Normalize, self ).__init__( trainable=trainable, **kwargs )
    self.shape = shape
      
  def build( self, input_shape ):
    weight_shape = self.shape or input_shape[-1:]

    self.means = self.add_weight(
      shape=weight_shape,
      name='means',
      initializer=initializers.Constant( 0 ),
      trainable=False
    )
    self.std_quotients = self.add_weight(
      shape=weight_shape,
      name='std_quotients',
      initializer=initializers.Constant( 1 ),
      trainable=False
    )
      
  def set_params( self, means, standard_deviations ):
    self.set_weights([
      np.asarray( means ),
      1./np.asarray(standard_deviations)
    ])
      
  def call( self, inputs ):
    # Subtract mean and MULTIPLY by standard deviation quotients
    # Multiply is used because the quotient is 1/std
    return ( inputs - self.means ) * self.std_quotients