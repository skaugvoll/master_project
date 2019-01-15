import functools
import yaml
import sys
import os

from .utils import custom_yaml_loader

class Config:

  def __init__( self ):

    # Move all variables from class to instance for convenience when dumping
    cls = self.__class__
    for key in vars( cls ):
      if key.startswith( '__' ):
        continue
      value = getattr( cls, key )
      if callable( value ):
        continue 
      setattr( self, key, value )

  @staticmethod
  def from_yaml( path, override_variables={} ):
    '''
    Load config from .yml format
    '''
    config = Config()
    with open( path, 'r' ) as f:
      data = custom_yaml_loader.load_yaml( f, override_variables )
    for k,v in data.items():
      setattr( config, k, v )
    return config 


  def get( self, key, default=None ):
    return getattr( self, key, default )

  def to_yaml( self, path ):
    '''
    Dumps the config to .yml format
    '''
    data = vars( self )
    with open( path, 'w' ) as f:
      yaml.dump( data, f )


  def pretty_print( self, key=None, buff=sys.stdout ):
    '''
    Writes the config or parts of the config in yaml
    notation to the specified buffer.
    '''
    value = vars( self ) if key is None else getattr( self, key )
    if type( value ) == str:
      buff.write( value+'\n' )
    else:
      yaml.dump( value, buff, default_flow_style=False, indent=2 )


if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser( description='Output single config values' )

  parser.add_argument( '-f', '--file',
    required = True,
    help     = 'Path to config file in .yml format'
  )
  parser.add_argument( '-k', '--key',
    help     = 'Key to retrieve, will print entire config if not specified'
  )
  args, _ = parser.parse_known_args()

  # Load config file
  config = Config.from_yaml( args.file )

  # Print just single key if specified
  config.pretty_print( key=args.key )
