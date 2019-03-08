from keras.layers import Layer, Input, Dense
from keras import initializers
from keras.models import Model

import numpy as np


class Normalize( Layer ):
    
  def __init__( self, shape=None, trainable=False, **kwargs ):
    '''
    Creates a new Object / class that can be used as tensor in Keras layer(s),
    and makes sure that the tensor is trainable aka can change value, aka traianble weights
    url: keras.io/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers

    :param shape:
    :param trainable:
    :param kwargs:
    '''
    super( Normalize, self ).__init__( trainable=trainable, **kwargs )
    self.shape = shape
      
  def build( self, input_shape ):
    '''
    This is where you will define your weights. This method must set self.built = True at the end,
      which can be done by calling super([layer], self).build()


    NOTE: this method does not set self.built = True, and the LSTM stil works, why ?  


    :param input_shape:
    :return:
    '''
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
    '''
    This is where the layer's logic lives. Unless you want your layer to support masking, you only have to care about the first argument passed to call: the input tensor
    :param inputs:
    :return:
    '''
    # Subtract mean and MULTIPLY by standard deviation quotients
    # Multiply is used because the quotient is 1/std
    return ( inputs - self.means ) * self.std_quotients