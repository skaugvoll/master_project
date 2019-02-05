import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: pass
import contextlib
import argparse 
import math

import pandas as pd 
import numpy as np 

from src.config import Config 
from src.utils import zip_utils
from src.utils import csv_loader
from src import models
from src import axivity

# VALID FORMATS FOR INPUT FILE
_CSV_FORMAT = 'csv'
_CWA_FORMAT = 'cwa_7z'
_FORMATS = [_CSV_FORMAT, _CWA_FORMAT]

parser = argparse.ArgumentParser( description='Runs a model in inference mode over a single subject' )

parser.add_argument( '-c', '--config',
  required = True,
  help     = 'Path to model config'
)
parser.add_argument( '-f', '--file',
  help    = 'Use a precomputed, timesynched file if provided'
)
parser.add_argument( '-F', '--format', 
  choices = _FORMATS,
  default = _CWA_FORMAT,
  help    = 'Format to the the input file should be in'
)
parser.add_argument( '-w', '--working-dir',
  help     = 'Which directory to use while working, data will be unzipped here and temporary files will be created. \
              It will be deleted afterwards'
)
parser.add_argument( '-m', '--model-dir',
  help     = 'Model directory that will be used when parsing config to e.g. locating weights. Will use\
              the same directory as config if not provided'
)
parser.add_argument( '-n', '--name',
  help     = 'Optional name of job. Otherwise, the basename of the input file will be used'
)
parser.add_argument( '-o', '--output-dir',
  required = True,
  help     = 'Directory where output will be stored. It will be nested under the basename of the input zip file'
)
parser.add_argument( '--chunk-size',
  type     = int,
  default  = 20000,
  help     = 'Number of rows of csv to reach in each chunk'
)

parser.add_argument( '--whole-days',
  action   = 'store_true',
  help     = 'Whether to only predict across whole days'
)
parser.add_argument( '--max-days',
  type     = int,
  default  = 6,
  help     = 'Max number of days after first midnight to predict (only relevant if whole days are used)'
)

parser.add_argument( '--plot-daily-overview',
  action   = 'store_true',
  help     = 'Whether a daily activity plot should be generated'
)
parser.add_argument( '--plot-uncertainty-thresh',
  type     = float,
  default  = 0.4,
  help     = 'Confidence threshold for daily-activity plot (only relevant if plot is enabled)'
)


@contextlib.contextmanager
def get_csv_file( args ):
  '''
  Returns a context for getting csv file, either by a directly specified one
  or through unzipping a provided zip file and synching it with axivity's software
  '''

  # If a precomputed, timesynched file is available -> use it
  if args.format == _CSV_FORMAT:
    if not os.path.exists( args.file ):
      raise RuntimeError( 'Provided presynched file "%s" does not exist'%args.file )
    job_name = args.name or os.path.splitext( os.path.basename( args.file ))[0]
    yield job_name, args.file

  # If a path to a zip-file is provided -> unzip, synch and use it
  elif args.format == _CWA_FORMAT:
    # Make sure zipfile exists
    if not os.path.exists( args.file ):
      raise RuntimeError( 'Provided zip file "%s" does not exist'%args.file ) 
    # Make sure that a working directory for unzipping and time synching also exists
    if not args.working_dir:
      raise RuntimeError( 'A working directory ("-w <directoy name.") must be specified when using --zip-file' )
    if not os.path.exists( args.working_dir ):
      raise RuntimeError( 'Provided working directory "%s" does not exist'%args.working_dir )

    # Unzip contents of zipfile
    job_name = args.name or os.path.splitext( os.path.basename( args.file ))[0]
    unzip_to_path = os.path.join( args.working_dir, os.path.basename( args.file ))
    with zip_utils.zip_to_working_dir( args.file, unzip_to_path ) as subject_dir:
      # Apply omconvert and timesync to join thigh & back .cwa files into a single .csv
      with axivity.timesynched_csv( subject_dir ) as synched_csv:
        yield job_name, synched_csv


  # Either a precomputed file or a timesynched file must be provided
  else:
    raise RuntimeError( 'Invalid format "%s"'%args.format )




def main( args ):

  
  # Get model directory (so we get the right weights etc)
  model_dir = args.model_dir or os.path.dirname( args.config )
  # Load config
  config = Config.from_yaml( args.config, override_variables={'MODEL_DIR':model_dir} )
  # Create output directory if it does not exist
  if not os.path.exists( args.output_dir ):
    os.makedirs( args.output_dir )

  # Get csv file for prediction
  with get_csv_file( args ) as (name, synched_csv):
    # Return if no csv file was found
    if synched_csv is None: return
    print( 'Got synched csv file:', synched_csv )
    # SYNCEHD_CSV IS NOT A OBJECT OR OPEN FILE, IT'S A PATH TO THE CSV FILE
    input("STOP ME NOW.... ctrl + c")
    
    # Read csv files in chunks
    columns = [ 'timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z' ]
    if args.whole_days:
      # Use custom loader that scans to first midnight if --whole-days is enabled
      dataframe_iterator = csv_loader.csv_chunker( synched_csv, args.chunk_size, ts_index=0,
                                                   columns=columns, n_days=args.max_days )
    else:
      # Otherwise, just load with pandas 
      dataframe_iterator = pd.read_csv( synched_csv, header=None, chunksize=args.chunk_size,
                                                     names=columns, parse_dates=[0] )

    # Create model
    print( 'Creating model' )
    model_name = config.MODEL['name']
    model_args = dict( config.MODEL['args'].items(), **config.INFERENCE.get( 'extra_model_args', {} ))
    model = models.get( model_name, model_args )
    model.summary()

    # Do inference
    print( 'Begin predicting' )
    results = model.inference( dataframe_iterator, **config.INFERENCE.get('args', {}) )
    print( 'Finished predicting')

    # Save output as a hdf5 file
    prediction_save_path = os.path.join( args.output_dir, '%s_timestamped_predictions.h5'%name )
    results.to_hdf( prediction_save_path, key='data' )
    print( 'Predictions saved to:', prediction_save_path )

    # Make a plot of daily overview
    if args.plot_daily_overview:
      print( 'Generating daily overview plot')
      from src.utils import plot_daily_overview
      daily_overview_savepath = os.path.join( args.output_dir, '%s_daily_overview.pdf'%name )
      plot_daily_overview.plot_weekly_view( results, config.CLASSES, daily_overview_savepath, 
                                            uncertain_thresh=args.plot_uncertainty_thresh )
      print( 'Daily overview plot save to:', daily_overview_savepath )

if __name__ == '__main__':
  args,_ = parser.parse_known_args()
  main( args )