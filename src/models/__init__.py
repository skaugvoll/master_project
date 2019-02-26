

class HARModel:
  '''
  Main interface for models used
  '''
  def __init__( self, **kwargs ):
    '''
    Define your own keyword arguments that will be passed from a yaml file
    '''
    pass

  def train( self, train_data, valid_data=None, **kwargs ):
    '''
    Train the model. Usually, we like to split the data into training and validation
    by producing a dichotomy over the subjects. This means that 

    Inputs:
      - train_data: list<pd.DataFrame>
        A list of dataframes, intended to be used for model fitting. It's columns are:
          <back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label> 
        Where the first 6 corresponds to sensor data as floats and the last one is an 
        integer corresponding to the annotated class
      - valid_data: list<pd.DataFrame>
        Same as train_data, execpt usually a much shorter list.
      - **kwargs:
        Extra arguments found in the model's config
    '''
    raise NotImplementedError()

  def inference( self, dataframe_iterator, **kwargs ):
    '''
    Do inference on unlabeled data.

    Inputs:
      - dataframe_iterator: iterator<pd.DataFrame>
        A python generator object that yields chunks of a dataframe. We use a generator because
        a typical inference job might be defined over a week worth of data, which easily amounts
        to more than a gigabyte. There are no guarantee that the length of the dataframe will be
        the same for every yield, so keep that in mind and account for it if your model implementation
        has a requirement for this. The columns in the dataframe are:
          <timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z>
        Where the (back/thigh)_(x/y/z) are the same as in training, and timestamp is a series of 
        python datetime objects corresponding to when each sensor sample was taken.
      - **kwargs:
        Extra arguemnts found in the model's config

    Outputs:
      A single dataframe with columns:
        <timestamp, prediction, confidence>
      Where timestamp are the same as in input, prediction is an integer indicating the predicted
      class and confidence is a float indicating the confidence of said class (just set it to 1.0) 
      if your model does not support confidence. The sample rate of the output does not have to match
      the sample rate of the input. For instance, you can take 50hz input, partition it into windows
      of 5 sec (0.2hz) and produce a single prediction for each window. You should, however, try to
      make sure that the sampling of output is even (e.g. 00:00, 00:05, 00:10, ... etc) and the timestamp
      of a window should (for the sake of convention) mark the end of said window.



    '''
    raise NotImplementedError()

  def summary( self ):
    print( '%s : No summary available'%self )



def get( name, model_args, make_growth_mode_session=True ):
  if not name in _MODELS:
    raise ValueError( 'Model %s is not in the list of available models'%name )
  Model = _MODELS[ name ]

  # Make a fresh session and set memory allocation in growth mode if enabled
  # This greatly saves on gpu memory
  if make_growth_mode_session:
    from keras import backend as K 
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session( K.tf.Session(config=cfg) )

  return Model( **model_args )



from .two_sensor_lstm import TwoSensorLSTM
from .one_sensor_lstm import OneSensorLSTM
from .snt_rfc import HARRandomForrest
from .testLSTM import LSTMTEST

_MODELS = {
  'TWO_SENSOR_LSTM' : TwoSensorLSTM,
  'ONE_SENSOR_LSTM' : OneSensorLSTM,
  'RFC': HARRandomForrest,
  'LSTMTEST': LSTMTEST
}