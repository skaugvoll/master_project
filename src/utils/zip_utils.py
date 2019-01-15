import subprocess
import contextlib
import os

@contextlib.contextmanager
def zip_to_working_dir( subject_zip_path, unzip_to_path, clean_up=True, 
                                  zip_back=False, return_inner_dir=True ):
  '''
  Context for unzipping and zipping back data from drive
  '''
  subject_dir = None
  try:
    # Unzip and yield control
    subject_dir = unzip_subject_data( subject_zip_path, unzip_to_path, return_inner_dir=return_inner_dir )
    yield subject_dir
    # Zip and store
    if zip_back:
      zip_subject_data( subject_dir, subject_zip_path )

  finally:
    # Clean up working dir 
    if clean_up and not subject_dir is None:
      clean_up_working_dir( unzip_to_path )


def unzip_subject_data( subject_zip_path, unzip_to_path, return_inner_dir=True ):

  # Make working directory for this subject. It should not already exist
  assert not os.path.exists( unzip_to_path ), 'Error: subject dir "%s" already exists'%unzip_to_path

  # Call 7z program to unzip. Note: this might not be available @SamuelX
  command = ['7z', 'x', subject_zip_path, '-o'+unzip_to_path ]
  print( 'Unzipping:', ' '.join( command ))
  ret = subprocess.call( command )
  assert ret == 0, 'Received non-zero (%s) return code when unzipping'%ret 

  if not return_inner_dir:
    return unzip_to_path

  # If requested, try to return an inner dir
  dir_content = os.listdir( unzip_to_path )
  # It should only be a single folder 
  assert len( dir_content ) == 1, 'Archive did not contain exactly one folder (%s)'%dir_content
  unzip_to_path = os.path.join( unzip_to_path, dir_content[0] )
  assert os.path.isdir( unzip_to_path ), 'Archive "%s" did not contain a folder'%subject_zip_path

  return unzip_to_path 


def zip_subject_data( subject_dir, subject_zip_path ):
  '''Zips the content of the subject directory and loads it back to where it came from'''
  command = ['7z', 'a', subject_zip_path, subject_dir]
  print( 'Zipping:', ' '.join( command ))
  ret = subprocess.call( command )
  assert ret == 0, 'Received non-zero (%s) return code when zipping'%ret

  return subject_zip_path


def clean_up_working_dir( unzip_to_path ):
  '''Cleans up the subject directory in the working directory'''
  command = ['rm', '-rf', unzip_to_path]
  ret = subprocess.call(command)
  assert ret == 0, 'Received non-zero (%s) return code when removing subject directory'%ret 

