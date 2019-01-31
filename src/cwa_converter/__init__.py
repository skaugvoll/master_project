import os
from axivity import find_back_cwa_file, find_thigh_cwa_file

def run_cwa_temp_convert(input_cwa, output_csv_path=None):
    '''
    CMD to use program:
    CWA <filename.cwa>
        [-s:accel|-s:gyro]
        [-f:csv|-f:raw|-f:wav]
        [-v:float|-v:int]
        [-t:timestamp|-t:none|-t:sequence|-t:secs|-t:days|-t:serial|-t:excel|-t:matlab|-t:block]
        [-no<data|accel|gyro|mag>]
        [-light]
        [-temp]
        [-batt[v|p|r]]
        [-events]
        [-amplify 1.0]
        [-start 0]
        [-length <len>]
        [-step 1]
        [-out <outfile>]
        [-blockstart 0]
        [-blockcount <count>]

    :param input_cwa:
    :param output_wav_path:
    :param output_csv_path:
    :return:
    '''

    cwa_convert_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cwa-convert")

    shell_command = [cwa_convert_script, input_cwa, '-temp']



    if output_csv_path is not None:
        print("CSV file will be output to", output_csv_path)
        shell_command += ['-out', output_csv_path]

    if not os.path.exists( omconvert_script ):
        print("Did not find a compiled version of cwa_convert. "
              "Building cwa_convert from source. This may take a while.")
        cwa_convert_directory = os.path.join(os.path.dirname(cwa_convert_script))
        make_cwa_convert_call = ["make", "-C", cwa_convert_directory]
        subprocess.call(make_cwa_convert_call)

    print( shell_command )
    subprocess.call(shell_command)
    print("cwa-convert finished.")

def convert_cwas_to_csv_with_temp( subject_dir, out_dir=None ):

  # Default to exporing to same directory as where the subject .cwa files are stored
  out_dir = out_dir or subject_dir

  # Find thigh and back .cwa files
  back_cwa = find_back_cwa_file( subject_dir )
  thigh_cwa = find_thigh_cwa_file( subject_dir )
  print( 'Found back_cwa: ', back_cwa )
  print( 'Found thigh_cwa:', thigh_cwa )

  # TODO: Consider whether the .md5 checksum often found in the zipped files should be used

  # Run conversion and synchronization program
  synched_csv = 1
  # synched_csv = timesync_from_cwa( back_cwa, thigh_cwa, out_dir=out_dir, clean_up=True )

  return synched_csv
