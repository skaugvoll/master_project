from keras.layers import Input, Conv1D, Concatenate, Dropout, Activation,\
                         Dense, Add, BatchNormalization, MaxPool1D
from keras.models import Model

from .  layers.conv1d_transpose import Conv1DTranspose



def Unet( config ):

  ipt = Input( shape=( config.SEQUENCE_LENGTH, config.NUM_CHANNELS ))

  net = Conv1D( 16, 7, padding='same', activation='relu' )( ipt )

  contractions = []

  # --- Contracting part

  for layer in config.LAYERS:

    net = Conv1D( layer['units'], 3, padding='same', activation='relu' )( net )

    if layer.get( 'contract', False ):
      net = MaxPool1D( pool_size=2 )( net )
      # Add to contractions so it can be joined on the way up
      contractions.append( net )

    if layer.get( 'dropout', False ):
      net = Dropout( layer['dropout'] )( net )


  # --- Expanding part
  i = -1 # Counter for joining with the correct layer on the opposite side of contractions

  for layer in reversed( config.LAYERS ):

    if layer.get( 'contract' ):
      # Scale up and concatenate with other side of the U
      net = Conv1DTranspose( layer['units'], 2, strides=2, padding='same' )( net )
      net = Concatenate()([ net, contractions[i] ])
      i -= 1

    # Apply convolution
    net = Conv1D( layer['units'], 3, padding='same', activation='relu' )( net )


  # --- Map to number of classes
  net = Conv1D( config.NUM_OUTPUTS, 1, padding='same', activation='relu' )( net )

  # --- Compile model
  model = Model( inputs=inputs, outputs=outputs )
  model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

  return model 




