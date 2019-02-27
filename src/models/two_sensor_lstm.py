import collections
import os

import pandas as pd
import numpy as np 

from keras.layers import Input, Concatenate, Dropout, Activation, Dense, Add, Bidirectional, BatchNormalization
from keras.models import Model

from . import HARModel
from ..layers.lstm import LSTM
from ..layers.normalize import Normalize
from ..layers.residual_addition import residual_addition
from ..utils import data_encoder
from ..utils import normalization
from ..utils import csv_loader
from ..callbacks import get_callback


class TwoSensorLSTM( HARModel ):

  def __init__( self, **params ):

    self.back_layers     = params.pop( 'back_layers' )
    self.thigh_layers    = params.pop( 'thigh_layers' )
    self.classes         = params.pop( 'classes' ) 
    self.sequence_length = params.pop( 'sequence_length', None ) 
    self.batch_size      = params.pop( 'batch_size', None ) 
    self.gpu             = params.pop( 'gpu' )
    self.stateful        = params.pop( 'stateful' ) 
    self.bidirectional   = params.pop( 'bidirectional' ) 
    self.output_dropout  = params.pop( 'output_dropout' ) 
    self.batch_norm      = params.pop( 'batch_norm' ) 

    self.encoder = data_encoder.DataEncoder( self.classes )
    self.num_outputs = self.encoder.num_active_classes


    # Build network
    self.build()




  def train( self,
      train_data,
      valid_data=None,
      callbacks=[],
      epochs=10,
      batch_size=None,
      sequence_length=None,
      back_cols=['back_x', 'back_y', 'back_z'],
      thigh_cols=['thigh_x', 'thigh_y', 'thigh_z'],
      label_col='label'
      ):

    # back_cols = ['bx', 'by', 'bz']
    # thigh_cols = ['tx', 'ty', 'tz']
    # label_col  = 'label'

    # Make batch_size and sequence_length default to architecture params
    batch_size = batch_size or self.batch_size
    sequence_length = sequence_length or self.sequence_length

    # Compute mean and standard deviation over training data
    back_means, back_stds = normalization.compute_means_and_stds( train_data, back_cols )
    thigh_means, thigh_stds = normalization.compute_means_and_stds( train_data, thigh_cols )
    # Update normalize layers in model with means and stds. This will bake normalization into the model weights
    self.norm_back.set_params( back_means, back_stds )
    self.norm_thigh.set_params( thigh_means, thigh_stds  )

    # Get design matrices of training data
    train_x1 = self.get_features( train_data, back_cols, batch_size=batch_size, sequence_length=sequence_length )
    train_x2 = self.get_features( train_data, thigh_cols, batch_size=batch_size, sequence_length=sequence_length )

    train_y  = self.get_labels( train_data, label_col, batch_size=batch_size, sequence_length=sequence_length )

    # Get design matrix of validation data if provided
    if valid_data is not None:
      validation_data = (
        [ self.get_features( valid_data, back_cols, batch_size=batch_size, sequence_length=sequence_length ),
          self.get_features( valid_data, thigh_cols, batch_size=batch_size, sequence_length=sequence_length ) ],
        self.get_labels( valid_data, label_col, batch_size=batch_size, sequence_length=sequence_length )
      )
    else:
      validation_data = None

    # Get callbacks for training
    # this uses the same logic as importing and running models, we export the available callbacks trough the __init__.py
    # from the root/src/callbacks module, which must be imported, thus in order to use this, the user must know what
    #   callbacks are avilable in the callback module
    callbacks = [ get_callback( cb['name'], **cb['args'] ) for cb in callbacks ]

    # Compile model AKA Configures the model for training.
    self.model.compile( loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'] )


    # I think the fit method extracts batches of specific (512) length automatically
    #  since it's build with the Input assumption,
    # and if there is some examples left that does not fill the full batch size, if NOT statefull, it runs the smaller batch
    #     if STATEFUL it ignores the examples that does not fill the batch
    # Begin training and return history once done
    return self.model.fit( 
      x=[train_x1, train_x2], y=train_y,
      validation_data=validation_data,
      batch_size=batch_size,
      epochs=epochs,
      shuffle=not self.stateful,
      callbacks=callbacks
    )


  def evaluate( self,
      dataframes,
      batch_size=None,
      sequence_length=None,
      back_cols=['back_x', 'back_y', 'back_z'],
      thigh_cols=['thigh_x', 'thigh_y', 'thigh_z'],
      label_col='label'
      ):

    # Make batch_size and sequence_length default to architecture params
    batch_size = batch_size or self.batch_size
    sequence_length = sequence_length or self.sequence_length

    # Get design matrices of training data
    x1 = self.get_features( dataframes, back_cols, batch_size=batch_size, sequence_length=sequence_length )
    x2 = self.get_features( dataframes, thigh_cols, batch_size=batch_size, sequence_length=sequence_length )

    y = self.get_labels( dataframes, label_col, batch_size=batch_size, sequence_length=sequence_length )

    return self.model.evaluate(
      x=[x1, x2],
      y=y,
      batch_size=batch_size,
    )


  def predict_on_one_window( self, window ):

    # todo: the params from self object includes the actual layer objects etc,
    # thus we might have to pass in the original config object and its values, as when we instntiate the model
    # to get the correct config values for creating the new model for predication on another bathc size

    # The idea here is that we want to predict on only one window aka (1, seq_length, features)
    # Thus we should create another two_sensor_lstm object
    # instansiate the new object with the same values as this, AND only change the batch_size property

    # TODO find out shape of windows used and reshape the window ?
    print(window, "\n", window.shape)


    params = self.__dict__
    params['batch_size'] = 1
    params['sequence_length'] = 1 # TODO change to be seq_length of window!


    # for k,v in params.items():
    #   print(k, v)

    predict_model = TwoSensorLSTM(**params)
    predict_model.model.set_weights(self.model.get_weights())
    print(predict_model.model.summary())

    x1 = predict_model.get_features([window], ['bx', 'by', 'bz'], batch_size=1, sequence_length=1)
    x2 = predict_model.get_features([window], ['tx', 'ty', 'tz'], batch_size=1, sequence_length=1)

    # print("X1: \n", x1, "\n", x1.shape)

    return predict_model.model.predict([x1, x2], batch_size=1)


  def inference( self, dataframe_iterator,
      batch_size=None,
      sequence_length=None,
      weights_path=None,
      timestamp_col='timestamp',
      back_cols=['back_x', 'back_y', 'back_z'],
      thigh_cols=['thigh_x', 'thigh_y', 'thigh_z']):

    # Make sure that valid weights are provided
    if not weights_path:
      raise ValueError( 'Weights path %s not specified'%weights_path )
    if not os.path.exists( weights_path ):
      raise ValueError( 'Weights path %s does not exist'%weights_path )

    print( 'Loading weights from "%s"'%weights_path )
    self.model.load_weights( weights_path )

    # timestamp_col = 'timestamp'
    # back_cols  = ['back_x', 'back_y', 'back_z']
    # thigh_cols = ['thigh_x', 'thigh_y', 'thigh_z']

    batch_size = batch_size or self.batch_size
    sequence_length = sequence_length or self.sequence_length
    # Wrap dataframe iterator so that it yields batches of appropriate size
    batch_iterator = csv_loader.batch_iterator( dataframe_iterator, batch_size=batch_size*sequence_length,
                                                                    allow_incomplete=not self.stateful )
    # Initialize empty list for inference output
    timestamps  = []
    predictions = []
    confidences = []

    # Begin processing batches
    for batch in batch_iterator:

      # Extract back and thigh sensor data
      batch_x1 = batch[ back_cols ].values.reshape( -1, sequence_length, len( back_cols ))
      batch_x2 = batch[ thigh_cols ].values .reshape( -1, sequence_length, len( thigh_cols ))

      # Predict on data
      raw_predictions = self.model.predict_on_batch( [batch_x1, batch_x2] )

      # Store the highest confidence
      confidences.append( raw_predictions.max( axis=1 ))

      # Decode argmaxes
      predictions.append( self.encoder.one_hot_decode( raw_predictions ))

      # Use the last timestamp in each sequence
      timestamps.append( batch[ timestamp_col ].values[ sequence_length-1::sequence_length ] )

    # Return a single dataframe
    return pd.DataFrame({
      'timestamp' : np.concatenate( timestamps ),
      'prediction': np.concatenate( predictions ),
      'confidence': np.concatenate( confidences )
    })


  def get_features( self, dataframes, columns, batch_size=None, sequence_length=None ):
    '''
    NOTE: The code was documented using the 4000181.7z
          after adding labels 1,2,3 for which model to use,
          and removing the rows where LABEL was not set

    This is a exciting and interesting function,
    this actually creates the batches and sequence length and all the good stuff I need to understand.

    :param dataframes: [df1, df2,..., dfn]
    :param columns: ["colName",..., "colName"]
    :param batch_size: Integer
    :param sequence_length: Integer
    :return:
    '''

    sequence_length = sequence_length or self.sequence_length

    # print("Batch size :: sequence length\n", batch_size, sequence_length)
    # >> 512, 250
    # print("Len dataframes: ", len(dataframes))
    # >> 1
    # print("Dataframes[0].head(2)")
    # >>
    #                               bx        by        bz       tx        ty        tz  label
    # timestamp
    # 2017-09-19 18:31:09.354 -0.046875 -0.078125  0.953125 -0.28125 -0.109375 -0.984375      1
    # 2017-09-19 18:31:09.374 -0.046875 -0.078125  0.953125 -0.28125 -0.109375 -0.984375      1

    # print("LEN dataframes[0] :: len dataframes[0] % seq_lenght", "\n", \
    #  len(dataframes[0]), len(dataframes[0]) % sequence_length)
    # >> 598807 57

    # NB THE len(dataframe) - len(dataframe) % sequence_length) is what assures that the reshape is valid and can be done!
    X = np.concatenate([
        dataframe[columns].values[ : (len(dataframe) - len(dataframe) % sequence_length) ] for dataframe in dataframes
      ]) #.reshape( -1, sequence_length, len(columns) )

    # print("X after concat:\n ", X.shape, "\n", X)
    # >> (598750, 3)
    #
    # [
    #   [-0.046875 - 0.078125  0.953125]
    #   [-0.046875 - 0.078125  0.953125]
    #   [-0.046875 - 0.078125  0.953125]
    #   ...
    #   [-0.8125 - 0.484375    0.6875]
    #   [-0.921875 - 0.25      0.25]
    #   [-0.90625 - 0.359375   0.421875]
    # ]

    X = X.reshape( -1, sequence_length, len(columns) )
    # print("X after reshape:\n", X.shape, "\n", X)
    # >> (2395, 250, 3)
    #
    # [
    #   [
    #     [-0.046875 - 0.078125  0.953125]
    #     [-0.046875 - 0.078125  0.953125]
    #     [-0.046875 - 0.078125  0.953125]
    #     ...
    #     [-0.046875 - 0.09375   0.953125]
    #     [-0.046875 - 0.078125  0.953125]
    #     [-0.046875 - 0.078125  0.9375]
    #   ]
    #   [
    #     [-0.046875 - 0.078125  0.953125]
    #     [-0.046875 - 0.078125  0.953125]
    #     [-0.046875 - 0.078125  0.953125]
    #     ...
    #     [-1. - 0.046875 - 0.0625]
    #     [-1. - 0.046875 - 0.046875]
    #     [-1. - 0.046875 - 0.046875]
    #   ]
    #   ...
    #   ...
    # ]

    # Here I'm guessing we have an array with 512 arrays, actually got 2395
    # where each of the 512 arrays, contain 250 arrays | actually got 250
    # where each of the 250 arrays contains x features | actually got 3
    # NB THE len(dataframe) - len(dataframe) % sequence_length) is what assures that the reshape is valid and can be done!


    if self.stateful:
      # No half-batches are allowed if using stateful. TODO: This should probably be done very differently
      batch_size = batch_size or self.batch_size
      X = X[ : (len(X) - len(X)%batch_size) ]

    return X 

  def get_labels( self, dataframes, column, batch_size=None, sequence_length=None ):

    sequence_length = sequence_length or self.sequence_length

    # Same as get_features(...) --> X.concatenate
    # except reshape is >> (2395, 250, 1)
    Y = np.concatenate([
        dataframe[column].values[ : len(dataframe) - len(dataframe)%sequence_length ]
        for dataframe in dataframes
      ]).reshape( -1, sequence_length )

    # print("Y shape:\n", Y.shape)
    # >> (2395, 250)


    # encoder.one_hot_encode() is a self made class
    #   takes in an array where each element is the most common target, for that input sequence
    #   The the array is:  [ s1, s2, s3, ... , sN], where s1,...sN is the most common target
    #   The function then returns a one hot 2D array with the one hot encoding for the array
    #   lets for instance say we only have 4 sequences with possible targets [1,2,3,4]
    #    the returned one hot vector is then for the imaginary array [1,1,3,1];
    #      [1,1,3,1] -> [
    #                     [1,0,0,0], ## sequence 1 one hot vector
    #                     [1,0,0,0], ## sequence 2 one hot vector
    #                     [0,0,1,0], ## sequence 3 one hot vector
    #                     [1,0,0,0]  ## sequence 4 one hot vector
    #                    ]
    #
    # .most_common(..) is from the Collections.Counter class
    #   List the n most common elements and their counts from the most
    #     common to the least.  If n is None, then list all element counts.
    #

    # Pick majority label in each sequence and One Hot encode
    Y = self.encoder.one_hot_encode( np.array([ collections.Counter( targets_sequence ).most_common(1)[0][0] for targets_sequence in Y ]))

    if self.stateful:
      # No half-batches are allowed if using stateful
      batch_size = batch_size or self.batch_size
      Y = Y[ : (len(Y) - len(Y)%batch_size)]

    return Y



  def build( self ):
    
    # Create input tensors; batch_size must be specified for stateful variant
    '''
    keras.layers.Input is used to instantiate a Keras tensor,

    Note! this does not create batches, only makes the input EXPECT a certain shape!

    Args:
      shape: A shape tuple (integers), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
      batch_size: optional static batch size (integer)
      batch_shape: Shape, including the batch size. For instance, shape = c(10,32) indicates that the expected input will be batches of 10 32-dimensional vectors.
                  alt. batch_shape=list(Null, 32) indicates batches of an arbitrary number of 32-dimensional vectors.

    :return:
    '''
    if self.stateful:
      # Create input with shape (batch_size, seq_length, features)
      ipt_back  = Input( batch_shape=[ self.batch_size, self.sequence_length, self.back_layers['inputs'] ])
      ipt_thigh = Input( batch_shape=[ self.batch_size, self.sequence_length, self.thigh_layers['inputs'] ])
    else:
      ipt_back  = Input( shape=[ self.sequence_length, self.back_layers['inputs'] ])
      ipt_thigh = Input( shape=[ self.sequence_length, self.back_layers['inputs'] ])

    # Create Input Normalization layers
    self.norm_back  = Normalize()
    self.norm_thigh = Normalize()

    # Create separate networks (LAYERS) for back sensor channels and thigh sensor channels
    back_net  = self.create_sub_net( self.norm_back( ipt_back ), self.back_layers['layers'] )
    thigh_net = self.create_sub_net( self.norm_thigh( ipt_thigh ), self.thigh_layers['layers'] )

    # print("back net: \n", back_net)
    # >> Tensor("bidirectional_1/add:0", shape=(512, 250, 32), dtype=float32)
    # What are back_net and thigh_net, they are instances of either;
    #    Bidirectional Layer or LSTM layer from Keras, or also known as Tensors
    # What are the layers axis ? thus, what is axis 2 ?
    # since the back_net is a tensor, it axis is its shape, aka;
    #   axis 0 = batch_size = 512
    #   axis 1 = sequence_lenght = 250
    #   axis 2 = features = 32
    #  Question : Why has it 32 features ???
    #   Answer: because we have specified in the config, that the (net / LSTM layer) should have 32 units!
    #           Keras.layers.LSTM :: units: Positive integer, dimensionality of the output space.


    # keras.layers.Concatenate(axis=-1)
    # Layer that concatenates a list of inputs.
    # It takes as input a list of tensors, all of the same shape except for the concatenation axis,
    # and returns a single tensor, the concatenation of all inputs.

    # Then combine the back_net tensor and thigh_net tensor into one tensor, on the second axis which is ?features?
    # we do this because we want to have a tensor where the input is (batch_size, seq_lenght, back_featrs + thigh_featrs)
    net = Concatenate(axis=2)([ back_net, thigh_net ])

    # print("NET: \n", net.shape, "\n", net)
    # >> (512, 250, 64)
    # Tensor("concatenate_1/concat:0", shape=(512, 250, 64), dtype=float32)

    # Apply dropout
    # Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time,
    # which helps prevent overfitting.
    net = Dropout( self.output_dropout )( net )

    # Add final LSTM LAYER
    # Here we pass in the expected number of outputs aka number of targets!
    # NOTE: basically DOWNSCALING the RNN to the final DENSE layers, so that it knows how many targets there are

    net = self.lstm_layer( units=self.num_outputs, return_sequences=False, 
                           stateful=self.stateful, gpu=self.gpu )( net )

    # Add softmax activation in the end
    # Keras.layers.Activation : Applies an activation function to an output.
    #     output shape :: Same shape as input.
    # TODO: is this the same as adding a DENSE layer?
    #   read some documentation, it kinda seems like it
    net = Activation( 'softmax' )( net )

    # Make model
    self.model = Model( inputs=[ipt_back, ipt_thigh], outputs=net )
    print(">>>>>>> BUILD COMPLETE")
    # TODO: Compile here?

  def create_sub_net( self, net, layers ):

    # Loop over all layers in definition
    for layer in layers:
      # Store current layer in case residual link
      prev_layer = net
      # Add LSTM Layer
      net = self.lstm_layer( units=layer['units'], return_sequences=True,
                             stateful=self.stateful, gpu=self.gpu )( net )
      # Apply dropout of specified
      if layer.get( 'dropout', False ):
        net = Dropout( layer['dropout'] )( net )
      # Apply residual connection to end of last layer if specified
      if layer.get( 'residual', False ):
        net = self.residual_addition( prev_layer, net )

    return net 

  def lstm_layer( self, *args, **kwargs ):
    if self.bidirectional:
      return Bidirectional( LSTM( *args, **kwargs ), merge_mode='sum' )
    else:
      return LSTM( *args, **kwargs )

  def residual_addition( self, l0, l1, activation='tanh' ):
    if self.batch_norm:
      return Activation(activation)(BatchNormalization()(Add()([l0,l1])))
    else:
      return Activation(activation)(Add()([l0,l1]))