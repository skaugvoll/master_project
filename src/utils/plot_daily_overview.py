import os
import matplotlib as mpl 
if os.getenv('MPL_USE_PDF_BACKEND'):
  mpl.use( 'PDF' )
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import pandas as pd
import numpy as np



def plot_weekly_view( data, classes, savepath, plot_title='Daily Activity', 
                      uncertain_thresh=None, ticks=12,
                      uncertain_color='black' , uncertain_class=-1 ):
  '''
  Generates a plot with one horizontal bar for each day, where activities 
  are color-coded along the bars
  
  # TODO: Rewrite to conform with classes used
  Input:
  - data
    A pandas DataFrame (or a filepath to a .csv file) with columns on the form:
    * <timestamp:datetime, label:integer> (old version)
    * <timestamp:datetime, prediction:integer, confidence:float> (new version)
  - classes
    A list of classes used. Each list entry should be a dictionary containing
    {
      - value: int        // The integer value corresponding to this class as found in :data:
      - name: string      // The human readable name corresponding to the class
      - plot: string|int  // If a string, it should contain three values separated 
                             by a colon with the format:
                              - <mpl_color_name:legend_color_name:legend_class_name>
                             If an int, it means that this class will not be plotted,
                             and all occurences of it in the data will instead be 
                             replaced by the class which :value: field matches the
                             integer.
    }
  - savepath
    Path to where the output plot should be saved
  - plot_title
    Title that will appear on the top of the plot
  - uncertain_thresh
    If set, it will set all predictions that have corresponding confidence below
    the given threshold to -1 (uncertain). This only works with the new format
    <timestamp, prediction, confidence>
  - ticks
    If set to an integer (default=12); as many ticks, evenly spaced in the 
    interval [00:00,...,24:00>. If set to a list of strings, use these instead,
    spaced evenly along each plotted bar's x-axis.
  - uncertain_color
    The color used for uncertain class
  - uncertain class
    The integer (value) used to denote uncertain class

  '''
  # Read csv files in case it is not a data frame
  if type( data ) == str:
    # TODO: Make option for reading HDF5 as well
    data = pd.read_csv( data, parse_dates=[0], header=None, names=['timestamp', 'prediction', 'confidence'] )
  # Otherwise -> Assume the input data should be a DataFrame

  # Dataframe where we will store plot data
  labelled_timestamp = pd.DataFrame()

  # Get timestamp column
  assert 'timestamp' in data.columns, 'Input data must contain a "timestamp" column'
  assert issubclass( data.dtypes['timestamp'].type, np.datetime64 ), '"timestamp" columns must be datetime'
  labelled_timestamp[ 'timestamp' ] = data[ 'timestamp' ]

  # Get label column
  if 'label' in data.columns:
    # Case: labels are already specified in the input (old solution)
    assert issubclass( data.dtypes[ 'label' ].type, np.integer ), '"label" column must be integer'
    labelled_timestamp[ 'label' ] = data[ 'label' ]
  else:
    # Case: data should contain a column for raw prediction (and their confidences)
    assert 'prediction' in data.columns, 'Input data must contain a prediction column'
    assert issubclass( data.dtypes['prediction'].type, np.integer ), '"prediction" column must be integer'
    if uncertain_thresh is None:
      # No confidence threshold specified -> Just use prediction
      labelled_timestamp[ 'label' ] = data[ 'prediction' ]
    else:
      # Use confidence threshold and set prediction's below it to -1
      assert 'confidence' in data.columns, 'Input data must contain a confidence column when a threshold is specified'
      assert data.dtypes[ 'confidence' ].type in [np.float16, np.float32, np.float64, np.float128], 'confidence column must be floating point'
      
      labelled_timestamp[ 'label' ] = np.where( data[ 'confidence' ] > uncertain_thresh, 
                                                data[ 'prediction' ], uncertain_class )
      # Add a class for 'Model uncertain'
      classes = [{ 'value': uncertain_class, 'name': 'model uncertain', 'plot':'%s:%s:model uncertain'%(uncertain_color,uncertain_color) }] + classes

  # Merge together classes (e.g. Cycling (sit) -> darkorange)
  class_value_to_color = _get_color_replace_dict( classes )
  labelled_timestamp = labelled_timestamp.replace({ 'label':class_value_to_color })

  # Find dates and unique days present in the predictions
  dates = labelled_timestamp['timestamp'].dt.date 
  days  = dates.drop_duplicates()

  # Pad before first datapoint with white so that the day is completely filled up
  # and equivalently after the last datapoint
  pad_before = _get_pad_data( labelled_timestamp, dates, days.iloc[0], before=True )
  pad_after  = _get_pad_data( labelled_timestamp, dates, days.iloc[-1], before=False )
  labelled_timestamp = pd.concat([ pad_before, labelled_timestamp, pad_after ], sort=False )
  dates = labelled_timestamp.loc[:, 'timestamp'].dt.date 

  # Initialize plot 
  n_days = len( days )
  fig, axes = plt.subplots( ncols=1, nrows=n_days, figsize=(20,2+1.5*n_days) )
  st = fig.suptitle( plot_title, fontsize='x-large' )
  if n_days == 1: axes = [axes] # If there is only one row, mpl will not return a tuple

  # Generate ticks if it is an integer
  if type( ticks ) == int:
    ticks = [ dt.strftime( '%H:%M' ) for dt in 
              pd.date_range( start=datetime(2000,1,1), end=datetime(2000,1,2), periods=ticks+1, closed='left' )]
  # Otherwise, ticks is assumed to already be a list of strings

  # Begin plotting
  for ax, day in zip( axes, days ):

    # Filter out data for current day
    data = labelled_timestamp.loc[ dates == day ].set_index( 'timestamp' )

    # Plot color bar
    cmap   = mpl.colors.ListedColormap( data.reset_index().label.tolist() )
    bounds = data.reset_index().index.tolist()
    norm   = mpl.colors.BoundaryNorm( bounds, cmap.N )
    color_bar = mpl.colorbar.ColorbarBase( ax, cmap=cmap, norm=norm, orientation='horizontal' )
    
    # Set ticks ticks along time axis
    ax.set_xticklabels( ticks )

  # Add legend
  legend = ' '.join( '%s: %s,'%x for x in _get_legend_items( classes ) )
  color_bar.set_label('Date: '+ days.iloc[0].__str__() + ' ' + legend )

  # Adjust y-position of title and subplots and pad the figure a bit
  # fig.tight_layout( pad=2 ) # NOTE: Tends to fail on OSX for .pdf backend and fall back to Agg Renderer
  st.set_y( 0.95 )
  fig.subplots_adjust( top=0.85 )

  # Save plot to disk if a savepath is provided; otherwise show it
  if savepath:
    plt.savefig( savepath )
  else:
    plt.show()
  plt.close()



def _get_color_replace_dict( classes ):
  '''
  Returns a dictionary that maps from class value (int)
  to a plot color (e.g. 'lightcyan')

  Classes should be a list of dictionaries, that contains at least
  { value->[int], plot->[string|int]}
  > If plot is an int, then the class will be replaced by the class
    with a value matchin the int
  > If plot is a string, then it should be formatted like:
    "matplotlibcolor:colorname:displayname"
  '''
  # Filter out classes that should be ignored (these are classes the model don't even use)
  used_classes = [ cl for cl in classes if type(cl.get('plot'))==str ]
  # Make sure classes are correctly formatted
  for cl in used_classes:
    plot = cl['plot']
    if len( plot.split(':') ) != 3:
      err = 'Class %s is badly formatted, it should be 3 substrigs separated by colons, but received %s'
      raise ValueError( err%( cl.get('name',cl.get('value')), plot))

  replace_dict = { cl['value']:cl['plot'].split(':')[0] for cl in used_classes }

  # Update replace dict with merged values
  for cl in ( cl for cl in classes if type( cl.get('plot')) == int ):
    merge_with = cl.get('plot')
    if not merge_with in replace_dict:
      err = 'Class %s supposed to be merged with %s which is not one of the used classes %s'
      raise ValueError( err%( cl.get('name',cl.get('value')), merge_with, list(replace_dict)))
    replace_dict[ cl['value'] ] = replace_dict[ cl['plot']]

  return replace_dict

def _get_legend_items( classes ):
  '''
  Returns a list of tuples on the form (color_name, class_name)
  '''
  legend_items = []
  for cl in classes:
    if type( cl.get('plot')) != str: continue
    _, color_name, class_name = cl.get('plot').split(':')
    legend_items.append(( color_name, class_name ))

  return legend_items


def _get_pad_data( labelled_timestamp, dates, day, before=True, pad_color='white' ):
  '''
  Generate padding data so that the datapoints for a given
  day spans a full 24 hour cycle
  '''
  # Determine spacing between each datapoint
  timestamps         = labelled_timestamp['timestamp']
  sampling_delta     = ((timestamps.iloc[-1] - timestamps.iloc[0])/(len(timestamps)-1))
  # Get first (inclusive) and last (exclusive) timestamp in pad data
  if before:
    first_timestamp = pd.Timestamp( day )
    last_timestamp  = timestamps.iloc[0]
  else:
    first_timestamp = timestamps.iloc[-1] + sampling_delta
    last_timestamp  = pd.Timestamp( day ) + pd.Timedelta( days=1 )

  pad_data = pd.DataFrame({'timestamp': pd.date_range(start=first_timestamp, freq=sampling_delta,
                                                      end=last_timestamp,  closed='left' )})
  pad_data['label'] = pad_color

  return pad_data 

