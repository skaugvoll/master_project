import keras 


def LSTM( units, gpu=False, **kwargs ):
  '''
  Uses optimized CuDNNLSTM if gpu is activated
  '''
  if gpu:
    layer = keras.layers.CuDNNLSTM( units=units, **kwargs )
  else:
    # Recurrent activation should default to sigmoid so that it
    # behaves similar to CuDNNLSTM. The normal default is hard 
    # sigmoid
    if not 'recurrent_activation' in kwargs:
      kwargs['recurrent_activation'] = 'sigmoid'
    layer = keras.layers.LSTM( units=units, **kwargs )

  return layer 