from __future__ import print_function

import os
import contextlib
import subprocess
import glob
import time

OMCONVERT_SCRIPT_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "omconvert", "omconvert")

TIMESYNC_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "timesync", "timesync")

@contextlib.contextmanager 
def timesynched_csv( subject_dir, out_dir=None, clean_up=True ):
    '''
    Context for converting .cwa files to .csv and timesyching
    while cleaning up everything afterwards
    '''
    synched_csv = None 

    try:
        # Convert & synch
        synched_csv = convert_cwas_to_csv( subject_dir, out_dir=out_dir )
        yield synched_csv

    finally:
        # Clean up synched .csv file
        if clean_up and synched_csv is not None and os.path.exists( synched_csv ):
            command = [ 'rm', synched_csv ]
            print( 'Removing synched .csv: ', ' '.join(command) )
            ret = subprocess.call( command )
            assert ret == 0, 'Received non-zero (%s) return code when removing .csv file'%ret 




def convert_cwas_to_csv( subject_dir, out_dir=None ):

  # Default to exporing to same directory as where the subject .cwa files are stored
  out_dir = out_dir or subject_dir 

  # Find thigh and back .cwa files
  back_cwa = find_back_cwa_file( subject_dir )
  thigh_cwa = find_thigh_cwa_file( subject_dir )
  print( 'Found back_cwa: ', back_cwa )
  print( 'Found thigh_cwa:', thigh_cwa )

  # TODO: Consider whether the .md5 checksum often found in the zipped files should be used

  # Run conversion and synchronization program
  synched_csv = timesync_from_cwa( back_cwa, thigh_cwa, out_dir=out_dir, clean_up=True )

  return synched_csv


def find_thigh_cwa_file( subject_dir, strict=True ):
  ''' Look for a thigh cwa file '''
  return _find_file( subject_dir, '*_T.cwa', strict=strict )

def find_back_cwa_file( subject_dir, strict=True ):
  ''' look for a back cwa file '''
  return _find_file( subject_dir, '*_B.cwa', strict=strict )

def _find_file( subject_dir, file_rgx, strict=True ):
  '''
  Searches for a file that matches a pattern in the given directory

  Inputs:
    - subject_dir
      A string containing a path to the directory that will be searched
    - file_rgx
      A string pattern to search for; "*" & "_" wildcards are allowed
    - strict=True
      Enforces that exactly one file is returned if set to true
  Output:
    The filepath to the matched files
  Throws:
    An exception if strict is set to true and the number of 
    matches is not exactly one
  '''
  # Find matching files
  match_string = os.path.join( subject_dir, file_rgx )
  try:
    # Only one file should match, if so -> return it
    matches = glob.glob( match_string )
    match, = matches
    return match
  
  except ValueError as e:
    if not strict:
      return None
    if 'not enough values to unpack' in str(e):
      raise Exception( 'No file found matching %s ' % match_string )
    if 'too many values to unpack' in str(e):
      raise Exception( '%s files matches %s' % ( len( matches ), match_string ))
    raise e



def run_omconvert(input_cwa, output_wav_path=None, output_csv_path=None):

    omconvert_script = OMCONVERT_SCRIPT_LOCATION

    shell_command = [omconvert_script, input_cwa]

    if output_wav_path is not None:
        print("WAV file will be output to", output_wav_path)
        shell_command += ['-out', output_wav_path]

    if output_csv_path is not None:
        print("CSV file will be output to", output_csv_path)
        shell_command += ['-csv-file', output_csv_path]

    if not os.path.exists( omconvert_script ):
        print("Did not find a compiled version of OMconvert. "
              "Building OMconvert from source. This may take a while.")
        omconvert_directory = os.path.join(os.path.dirname(omconvert_script))
        make_omconvert_call = ["make", "-C", omconvert_directory]
        subprocess.call(make_omconvert_call)

    print( shell_command )
    subprocess.call(shell_command)
    print("OMconvert finished.")


def timesync_from_cwa(master_cwa, slave_cwa, out_dir=None, out_file=None, clean_up=True):
    
    # Location of program used to synch back/thigh data
    timesync_script = TIMESYNC_LOCATION

    # Determine output directory
    out_dir = out_dir or os.path.dirname(master_cwa)
    assert os.path.exists( out_dir ), 'Output dir "%s" does not exist!' % out_dir 

    # Determine filepath of intermediary .wav files
    master_basename_without_extension = os.path.splitext(os.path.basename(master_cwa))[0]
    slave_basename_without_extension  = os.path.splitext(os.path.basename(slave_cwa))[0]
    master_wav = os.path.join( out_dir, master_basename_without_extension+'.wav' )
    slave_wav  = os.path.join( out_dir, slave_basename_without_extension+'.wav' )

    # Determine filepath of final output file
    timesync_output_file = out_file or master_basename_without_extension + "_" + slave_basename_without_extension + "_timesync_output.csv"
    timesync_output_path = os.path.join(out_dir, timesync_output_file)

    # These will be removed if clean_up is enabled
    intermediary_files = [master_wav, slave_wav]

    try:
        # Convert slave and master to .wav files
        print("Converting master and slave CWA files to intermediary WAV files")
        run_omconvert(master_cwa, output_wav_path=master_wav)
        run_omconvert(slave_cwa, output_wav_path=slave_wav)

        # Compile timesync program if not already done
        if not os.path.exists( timesync_script ):
            print("Did not find a compiled version of Timesync. "
                  "Building Timesync from source. This may take a while.")
            timesync_directory = os.path.dirname(timesync_script)
            make_call = ["make", "-C", timesync_directory]
            subprocess.call(make_call)

        # Execute timesync -> align slave output to master output w.r.t. time
        print("Running Timesync")
        subprocess.call([timesync_script, master_wav, slave_wav, "-csv", timesync_output_path])

        # RETURN the timesyched file which contains csv with:
        # - <timestamp, master_x, master_y, master_z, slave_x, slave_y, slave_z>
        return timesync_output_path

    except Exception as e:
        print( 'Encountered exception during conversion:', e )
        raise e 

    finally:
        # If enabeled -> remove intermediary files
        if clean_up:
            print("Removing intermediary files", intermediary_files)
            for f in intermediary_files:
                subprocess.call(["rm", f])

