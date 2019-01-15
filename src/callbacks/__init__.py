
from keras.callbacks import EarlyStopping, ModelCheckpoint

from .load_best_weights_reduce_lr import LoadBestWeigtsReduceLR
from .stop_on_overfit import Overfitting_callback


_CALLBACKS = {
  # Keras built-in callbacks
  'EarlyStopping' : EarlyStopping,
  'ModelCheckpoint' : ModelCheckpoint,
  # Custom callbacks
  'LOAD_BEST_WEIGHTS_REDUCE_LR' : LoadBestWeigtsReduceLR,
  'OVERFITTING_CALLBACK' : Overfitting_callback
}


def get_callback( callback_name, **callback_args ):

  if not callback_name in _CALLBACKS:
    raise ValueError( 'No callback called "%s" is registered; options: %s'%( callback_name, _CALLBACKS))

  return _CALLBACKS[ callback_name ]( **callback_args )

