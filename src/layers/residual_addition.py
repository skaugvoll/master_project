from keras.layers import BatchNormalization, Add, Activation



def residual_addition( l0, l1, batch_norm, activation='tanh' ):
  if batch_norm:
    return Activation(activation)(BatchNormalization()(Add()([l0,l1])))
  else:
    return Activation(activation)(Add()([l0,l1]))
