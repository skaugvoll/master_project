import functools
import pandas as pd
import scipy.signal


RESAMPLERS = {}
def get_resampler( name ):
  ''' Get a resampling function by name '''
  if not name in RESAMPLERS:
    raise Exception( 'No resampler with name {} is registered'.format(name) )
  return RESAMPLERS[name]

def get_resampler_names():
  ''' Get all resampling function names '''
  return sorted( RESAMPLERS )

def resampler( name=None ):
  ''' Register a resampler, optionally under a specific name '''
  def decorator( f ):
    RESAMPLERS[ name or f.__name__ ] = f
  return decorator


def resample_stream( resampler, dataframe_iterator, downsample_factor, discrete_columns ):
  '''
  Resample a stream of sensor data with a
  provided resampling function
  Inputs:
    - resampler : ( np.array[float], float ) => np.array[float]
      A resampling function that will be applied as a callback
    - dataframe_iterator : iterator[pd.DataFrame]
      A dataframe with a datetime index
    - downsample_factor : float
      Amount of downsampling to do. For instance, a value of
      2.0 will reduce the sample rate by half.
    - discrete_columns : list[string]
      Columns that should not be resampled by the resampling
      function. Instead, the closest value to the resampled
      index will be used.
  '''
  discrete_columns = discrete_columns or []

  for chunk in dataframe_iterator:
    # Get length of current window
    window_size = len( chunk )
    # Compute length of resampled window
    resampled_window_size = int( round( window_size / downsample_factor ))
    # Compute resampled index
    index = chunk.index
    start = index[0]
    end   = index[-1] + (index[-1]-index[0])/window_size
    resampled_index = pd.date_range( start, end, periods=resampled_window_size+1, closed='left' )
    # For all discrete columns, fetch the closest value to new index
    discrete_samples = index.searchsorted( resampled_index ).clip( 0, len(chunk)-1 )
    resampled_df = chunk[discrete_columns].iloc[ discrete_samples ]
    # Then set new index
    resampled_df.index = resampled_index
    # For the non-discrete columns, apply the resampler
    for column in set( chunk.columns ) - set( discrete_columns ):
      resampled_df[ column ] = resampler( chunk[column], downsample_factor )
    # Finally, make sure the resampled dataframe has the same column order
    yield resampled_df[ chunk.columns ]


@resampler( 'fourier' )
def resample_fourier( data, resample_factor ):
  new_length = int( round( len(data) / resample_factor ))
  return scipy.signal.resample( data, new_length )


@resampler( 'decimate' )
def resample_decimate( data, resample_factor ):
  if not resample_factor == int( resample_factor ):
    raise Exception( 'Resample factor must be an integer for decimate resampling' )
  res = scipy.signal.decimate( data, int(resample_factor), zero_phase=True )
  return res