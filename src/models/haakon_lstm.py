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


class HaakonLSTM( HARModel ):

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




  def train( self, train_data, 
      valid_data=None,
      callbacks=[],
      epochs=10,
      batch_size=None,
      sequence_length=None ):

    back_cols  = ['back_x', 'back_y', 'back_z']
    thigh_cols = ['thigh_x', 'thigh_y', 'thigh_z']
    label_col  = 'label'

    # Make batch_size and sequence_length default to architecture params
    batch_size = batch_size or self.batch_size
    sequence_length = sequence_length or self.sequence_length
    
    # Compute mean and standard deviation over training data
    back_means, back_stds   = normalization.compute_means_and_stds( train_data, back_cols )
    thigh_means, thigh_stds = normalization.compute_means_and_stds( train_data, thigh_cols )
    # Update normalize layers in model with means and stds. This will bake normalization into the model weights
    self.norm_back.set_params( back_means, back_stds )
    self.norm_thigh.set_params( thigh_means, thigh_stds  )

    # Get design matrices of training data
    train_x1 = self.get_features( train_data, back_cols, batch_size=batch_size, sequence_length=sequence_length )
    train_x2 = self.get_features( train_data, thigh_cols, batch_size=batch_size, sequence_length=sequence_length )
    train_y  = self.get_labels( train_data, label_col, batch_size=batch_size, sequence_length=sequence_length )
    # Get design matrix of validation data if provided
    if valid_data:
      validation_data = (
        [ self.get_features( valid_data, back_cols, batch_size=batch_size, sequence_length=sequence_length ),
          self.get_features( valid_data, thigh_cols, batch_size=batch_size, sequence_length=sequence_length ) ],
        self.get_labels( valid_data, label_col, batch_size=batch_size, sequence_length=sequence_length )
      )
    else:
      validation_data = None

    # Get callbacks for training
    callbacks = [ get_callback( cb['name'], **cb['args'] ) for cb in callbacks ]

    # Compile model
    self.model.compile( loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'] )

    # Begin training and return history once done
    return self.model.fit( 
      x=[train_x1, train_x2], y=train_y,
      validation_data=validation_data,
      batch_size=batch_size,
      epochs=epochs,
      shuffle=not self.stateful,
      callbacks=callbacks
    )


  def inference( self, dataframe_iterator,
      batch_size=None,
      sequence_length=None,
      weights_path=None ):

    # Make sure that valid weights are provided
    if not weights_path:
      raise ValueError( 'Weights path %s not specified'%weights_path )
    if not os.path.exists( weights_path ):
      raise ValueError( 'Weights path %s does not exist'%weights_path )

    print( 'Loading weights from "%s"'%weights_path )
    self.model.load_weights( weights_path )

    timestamp_col = 'timestamp'
    back_cols  = ['back_x', 'back_y', 'back_z']
    thigh_cols = ['thigh_x', 'thigh_y', 'thigh_z']

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

    sequence_length = sequence_length or self.sequence_length

    X = np.concatenate([
        dataframe[columns].values[ : (len(dataframe) - len(dataframe)%sequence_length) ]
        for dataframe in dataframes
      ]).reshape( -1, sequence_length, len(columns) )

    if self.stateful:
      # No half-batches are allowed if using stateful. TODO: This should probably be done very differently
      batch_size = batch_size or self.batch_size
      X = X[ : (len(X) - len(X)%batch_size) ]

    return X 

  def get_labels( self, dataframes, column, batch_size=None, sequence_length=None ):

    sequence_length = sequence_length or self.sequence_length

    Y = np.concatenate([
        dataframe[column].values[ : len(dataframe) - len(dataframe)%sequence_length ]
        for dataframe in dataframes
      ]).reshape( -1, sequence_length )

    # Pick majority label in each sequence and One Hot encode
    Y = self.encoder.one_hot_encode( np.array([
      collections.Counter( targets_sequence ).most_common(1)[0][0]
      for targets_sequence in Y
    ]))

    if self.stateful:
      # No half-batches are allowed if using stateful
      batch_size = batch_size or self.batch_size
      Y = Y[ : (len(Y) - len(Y)%batch_size)]

    return Y



  def build( self ):
    
    # Create input tensors; batch_size must be specified for stateful variant
    if self.stateful:
      ipt_back  = Input( batch_shape=[ self.batch_size, self.sequence_length, self.back_layers['inputs'] ])
      ipt_thigh = Input( batch_shape=[ self.batch_size, self.sequence_length, self.thigh_layers['inputs'] ])
    else:
      ipt_back  = Input( shape=[ self.sequence_length, self.back_layers['inputs'] ])
      ipt_thigh = Input( shape=[ self.sequence_length, self.back_layers['inputs'] ])

    # Create Input Normalization layers
    self.norm_back  = Normalize()
    self.norm_thigh = Normalize()

    # Create separate networks for back sensor channels and thigh sensor channels
    back_net  = self.create_sub_net( self.norm_back( ipt_back ), self.back_layers['layers'] )
    thigh_net = self.create_sub_net( self.norm_thigh( ipt_thigh ), self.thigh_layers['layers'] )

    # Then combine them into one
    net = Concatenate(axis=2)([ back_net, thigh_net ])
    # Apply dropout
    net = Dropout( self.output_dropout )( net )
    # Add final LSTM
    net = self.lstm_layer( units=self.num_outputs, return_sequences=False, 
                           stateful=self.stateful, gpu=self.gpu )( net )
    # Add softmax activation in the end
    net = Activation( 'softmax' )( net )
    # Make model
    self.model = Model( inputs=[ipt_back, ipt_thigh], outputs=net )
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