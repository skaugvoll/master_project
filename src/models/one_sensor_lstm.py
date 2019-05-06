import collections
import os

import pandas as pd
import numpy as np 

from keras.layers import Input, Concatenate, Dropout, Activation, Dense, Add, Bidirectional, BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix

from . import HARModel
from ..layers.lstm import LSTM
from ..layers.normalize import Normalize
from ..utils import data_encoder
from ..utils import normalization
from ..utils import csv_loader
from ..utils.ColorPrint import ColorPrinter
from ..callbacks import get_callback


class OneSensorLSTM( HARModel ):

  def __init__( self, **params ):

    self.layers     = params.pop( 'layers' )
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

    self.colorPrinter = ColorPrinter()
    self.test_ground_truth_labels = []
    self.predictions = []

    # Build network
    self.build()


  def save_model(self, path=False, model=False, weight=False):
    '''

    :param model_path: rel path from the "src/" including filename, excluding format. E.g  inside the src/ dir: "trained_models/twosensorlstm", makes a new directory /src/trained_models/twosensorlstm
    :return: path to saved model
    '''

    try:
      # check that the path exist, else create the directory to store the file
      if not os.path.exists(os.getcwd() + "/" + os.path.dirname(path)):
        os.mkdir(os.getcwd() + "/" + os.path.dirname(path))

      # # serialize model to JSON
      # model_json = self.model.to_json()
      # with open("{}.json".format(os.getcwd() + "/" + model_path), "w") as json_file:
      #   json_file.write(model_json)
      #
      # # serialize weights to HDF5
      # self.model.save_weights("{}.h5".format(os.getcwd() + "/" + model_path + '_weights'))
      # print("Saved model to disk")
      #
      self.saved_path = os.getcwd() + "/" + path
      status = ": "

      if model:
        status += "model saved, "
        self.model.save(os.getcwd() + "/" + path + ".h5")
      if weight:
        status += "weight saved,"
        self.model.save_weights(os.getcwd() + "/" + path + "_weights.h5")

      return self.saved_path + status
    except Exception as e:
      print("Could not save to disk, ", e)


  def train( self,
        train_data,
        valid_data=None,
        callbacks=[],
        epochs=10,
        batch_size=None,
        sequence_length=None,
        cols=['back_x', 'back_y', 'back_z'],
        label_col='label',
        shuffle=False
      ):

    # Make batch_size and sequence_length default to architecture params
    batch_size = batch_size or self.batch_size
    sequence_length = sequence_length or self.sequence_length

    # Compute mean and standard deviation over training data
    means, stds = normalization.compute_means_and_stds( train_data, cols )

    # Update normalize layers in model with means and stds. This will bake normalization into the model weights
    # self.norm.set_params( means, stds )

    # Get design matrices of training data
    train_x1 = self.get_features( train_data, cols, batch_size=batch_size, sequence_length=sequence_length )

    train_y = self.get_labels( train_data, label_col, batch_size=batch_size, sequence_length=sequence_length )

    if shuffle:
      np.random.seed(47)  # set the seed so that the shuffling is the same for all shuffles
      np.random.shuffle(train_x1)
      np.random.shuffle(train_y)


    # Get design matrix of validation data if provided
    if valid_data is not None:
      validation_data = (
        [
          self.get_features( valid_data, cols, batch_size=batch_size, sequence_length=sequence_length ),
          self.get_labels( valid_data, label_col, batch_size=batch_size, sequence_length=sequence_length )
        ]
      )
    else:
      validation_data = None

    # Get callbacks for training
    # this uses the same logic as importing and running models, we export the available callbacks trough the __init__.py
    # from the root/src/callbacks module, which must be imported, thus in order to use this, the user must know what
    #   callbacks are avilable in the callback module
    callbacks = [ get_callback( cb['name'], **cb['args'] ) for cb in callbacks ]

    # Compile model AKA Configures the model for training.
    #self.model.compile( loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'] )

    
    self.model.summary()
    # I think the fit method extracts batches of specifiec (512) length automatically
    #  since it's build with the Input assumption,
    # and if there is some examples left that does not fill the full batch size, if NOT statefull, it runs the smaller batch
    #     if STATEFUL it ignores the examples that does not fill the batch
    # Begin training and return history once done
    return self.model.fit( 
      x=train_x1, y=train_y,
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
      cols=['x', 'y', 'z'],
      label_col='label'
      ):

    # Make batch_size and sequence_length default to architecture params
    batch_size = batch_size or self.batch_size
    sequence_length = sequence_length or self.sequence_length

    # Get design matrices of training data
    x1 = self.get_features( dataframes, cols, batch_size=batch_size, sequence_length=sequence_length )
    y = self.get_labels( dataframes, label_col, batch_size=batch_size, sequence_length=sequence_length )

    loss, acc = self.model.evaluate(
      x=x1,
      y=y,
      batch_size=batch_size,
    )

    print("ACC: ", acc)
    return (loss, acc)


    # ret_labels = self.model.metrics_names
    # return_object = {}
    # for i, l in enumerate(ret_labels):
    #   return_object[l] = res[i]
    #
    # return return_object
    #
    # return self.model.evaluate(
    #   x=x1,
    #   y=y,
    #   batch_size=batch_size,
    # )



  def predict( self,
      dataframes,
      batch_size=None,
      sequence_length=None,
      cols=['x', 'y', 'z'],
      label_col='label'
      ):

    # Make batch_size and sequence_length default to architecture params
    batch_size = batch_size or self.batch_size
    sequence_length = sequence_length or self.sequence_length

    # Get design matrices of training data
    x1 = self.get_features( dataframes, cols, batch_size=batch_size, sequence_length=sequence_length )

    # Get labels as onehotencoding
    y = self.get_labels( dataframes, label_col, batch_size=batch_size, sequence_length=sequence_length )

    # Variables for evaluation, debugging, etc
    self.test_ground_truth_labels = y


    self.predictions = self.model.predict(
      x=x1,
      batch_size=batch_size,
    )

    return self.predictions, self.test_ground_truth_labels, self.calculate_confusion_matrix()

  def predict_on_one_window(self, window):
    '''

    :param window:
    :return:
    '''
    # self.compile()
    classification = self.model.predict(window, batch_size=1)
    prob = classification.max(axis=1)
    target = classification.argmax(axis=1)
    return target, prob

  def inference( self, dataframe_iterator,
      cols = ['x', 'y', 'z'],
      batch_size=None,
      sequence_length=None,
      weights_path=None):

    # Make sure that valid weights are provided
    if not weights_path:
      raise ValueError( 'Weights path %s not specified'%weights_path )
    if not os.path.exists( weights_path ):
      raise ValueError( 'Weights path %s does not exist'%weights_path )

    print( 'Loading weights from "%s"'%weights_path )
    self.model.load_weights( weights_path )

    timestamp_col = 'timestamp'

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
      batch = batch[ cols ].values.reshape( -1, sequence_length, len( cols ))

      # Predict on data
      raw_predictions = self.model.predict_on_batch( batch )
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
    print(self.colorPrinter.colorString('BUILDING LSTM...', color="yellow"), end='')

    if self.stateful:
      # Create input with shape (batch_size, seq_length, features)
      ipt = Input( batch_shape=[ self.batch_size, self.sequence_length, self.layers['inputs'] ])

    else:
      ipt = Input( shape=[ self.sequence_length, self.layers['inputs'] ])


    # Create Input Normalization layers
    self.norm = Normalize()

    # Hack for differantiating the input layer and the network
    # add a dropout that doesent do anything
    # net = Dropout(0.0)(ipt)

    # Create separate networks (LAYERS) for back sensor channels and thigh sensor channels
    for idx, layer in enumerate(self.layers['layers']):
      # Check if its the first layer, or intermidiate layer, and decide input to the new we're going to add
      if idx == 0:
        prev_layer = ipt
        layer_input = ipt
      else:
        prev_layer = net
        layer_input = net


      net = Bidirectional(
        LSTM(units=layer['units'],
            return_sequences=True,
            stateful=self.stateful,
            gpu=self.gpu
            ),
        merge_mode='sum')(layer_input)

      if layer.get( 'dropout', False ):
        net = Dropout( layer['dropout'] )( net )

      # Apply residual connection to end of last layer if specified
      if layer.get( 'residual', False ):
        net = self.residual_addition( prev_layer, net )



    # Apply dropout
    # Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time,
    # which helps prevent overfitting.
    net = Dropout( self.output_dropout )(net )

    # Add final LSTM LAYER
    # Here we pass in the expected number of outputs aka number of targets!
    # NOTE: basically DOWNSCALING the RNN to the final DENSE layers, so that it knows how many targets there are

    net = self.lstm_layer( units=self.num_outputs, return_sequences=False,
                           stateful=self.stateful, gpu=self.gpu )( net )

    # Add softmax activation in the end
    # Keras.layers.Activation : Applies an activation function to an output.
    #     output shape :: Same shape as input.
    # TODO: is this the same as adding a DENSE layer?
    out = Activation( 'softmax' )( net )
    # Make model
    self.model = Model(inputs=ipt, outputs=out )
    self.compile()

    print(self.colorPrinter.colorString(">>>>>>> BUILD COMPLETE <<<<<<<<", "green", bright=True))
    print(self.model.summary())

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

  def compile(self):
    self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


  def calculate_confusion_matrix(self):
      gt = self.test_ground_truth_labels # is one hot vector
      preds = self.predictions # is probability for each class

      gt2 = self.encoder.one_hot_decode(gt)
      gt = gt.argmax(axis=1)

      preds = preds.argmax(axis=1)

      labels_used = set(preds) # [0, n]

      labels = [self.encoder.name_lookup[i + 1] for i in labels_used]

      self.confusion_matrix = confusion_matrix(gt, preds)
      # self.print_cm(self.confusion_matrix, labels)

      return self.confusion_matrix


  def print_cm(self, cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
      print("%{0}s".format(columnwidth) % label, end=" ")
    print()

    # Print rows
    for i, label1 in enumerate(labels):
      print("    %{0}s".format(columnwidth) % label1, end=" ")
      for j in range(len(labels)):
        cell = "%{0}.1f".format(columnwidth) % cm[i, j]
        if hide_zeroes:
          cell = cell if float(cm[i, j]) != 0 else empty_cell
        if hide_diagonal:
          cell = cell if i != j else empty_cell
        if hide_threshold:
          cell = cell if cm[i, j] > hide_threshold else empty_cell
        print(cell, end=" ")
      print()