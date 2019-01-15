'''
Module for logging in the application

TODO: Make it possible to log to file or out or both, etc...
'''

import logging
import argparse
import inspect
import functools
import termcolor

# ---- Decorator for showing file and line number of logging statements

def log( prefix=None ):
  def decorator( f ):
    _prefix = prefix or f.__name__.upper()
    @functools.wraps( f )
    def decorated( msg, *args, **kwargs ):
      caller = inspect.getframeinfo( inspect.stack()[1][0] )
      msg = '[%s] %s:%s: %s'%( _prefix, caller.filename, caller.lineno, msg )
      f( msg, *args, **kwargs )
    return decorated
  return decorator



# ---- INTERFACE 

@log()
def debug( msg, *params ):
  # print( msg%params )
  termcolor.cprint( msg%params, 'cyan' )

@log()
def info( msg, *params ):
  print( msg%params )

@log()
def warning( msg, *params ):
  termcolor.cprint( msg%params, 'yellow' )

@log()
def error( msg, *params ):
  termcolor.cprint( msg%params, 'red' )

@log()
def critical( msg, *params ):
  termcolor.cprint( msg%params, 'red', attrs=['bold'] )

# Alias for warning
warn = warning


def test( message ):
  caller = inspect.getframeinfo( inspect.stack()[1][0] )
  print( "%s:%d - %s" % (caller.filename, caller.lineno, message))