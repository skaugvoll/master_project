'''
Keras callback for stopping training when overfitting is detected.

Author: Haakon Maloy
'''
import keras 
import warnings 
import numpy as np 
from keras import backend as K 

class Overfitting_callback(keras.callbacks.Callback):
  """
  Stop training when model overfits.

  # Arguments
      monitor: quantities to be monitored(list of at least two elements).
      patience: number of epochs with overfitting
          after which training will be stopped.
      verbose: verbosity mode.
      mode: one of {auto, min, max}. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `max`
          mode it will stop when the quantity
          monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
  """

  def __init__(self, monitor=['loss', 'val_loss'],
               patience=0, verbose=0, mode='auto'):
    super(Overfitting_callback, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.wait = 0
    self.stopped_epoch = 0

    if mode not in ['auto', 'min', 'max']:
      warnings.warn('Overfitting mode %s is unknown, '
                    'fallback to auto mode.' % mode,
                    RuntimeWarning)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor[1]:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less


  def on_train_begin(self, logs=None):
      # Allow instances to be re-used
      self.wait = 0
      self.stopped_epoch = 0
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def on_epoch_end(self, epoch, logs=None):
    train_loss = logs.get(self.monitor[0])
    val_loss = logs.get(self.monitor[1])
    if val_loss is None:
      warnings.warn(
          'Overfitting conditioned on metric `%s` '
          'which is not available. Available metrics are: %s' %
          (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
      )
      return
    # train_loss > val_loss
    if not self.monitor_op(train_loss, val_loss):
      self.wait = 0

    # train_loss < val_loss
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True


