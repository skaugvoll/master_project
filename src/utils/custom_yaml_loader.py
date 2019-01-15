from yaml.reader import *
from yaml.scanner import *
from yaml.parser import *
from yaml.composer import *
from yaml.resolver import *
from yaml.constructor import *
from yaml.nodes import *
from yaml.events import *

import os


class OverrideableComposer(Composer):

  def compose_node(self, parent, index):
    if self.check_event(AliasEvent):
      event = self.get_event()
      anchor = event.anchor
      if anchor not in self.anchors:
        raise ComposerError(None, None, "found undefined alias %r"
                            % anchor, event.start_mark)
      # this is the only change from the canonical Composer
      setattr(self.anchors[anchor], 'anchor_name', anchor)
      if anchor in self.config_override:
        return self.config_override[ anchor ]
      return self.anchors[anchor]
    event = self.peek_event()
    anchor = event.anchor
    if anchor is not None:
      if anchor in self.anchors:
        raise ComposerError("found duplicate anchor %r; first occurence"
                            % anchor, self.anchors[anchor].start_mark,
                            "second occurence", event.start_mark)
        
    self.descend_resolver(parent, index)
    if self.check_event(ScalarEvent):
      node = self.compose_scalar_node(anchor)
    elif self.check_event(SequenceStartEvent):
      node = self.compose_sequence_node(anchor)
    elif self.check_event(MappingStartEvent):
      node = self.compose_mapping_node(anchor)
    self.ascend_resolver()
    
    if anchor in self.config_override:
      self.unused_override.remove( anchor )
      return self.config_override[ anchor ]
    return node


class CustomYamlLoader(Reader, Scanner, Parser, OverrideableComposer, Constructor, Resolver):
  '''
  Override pyyaml's builtin loader so we can add
  a few convenient yaml document constructs
  '''
  def __init__(self, stream, config_override={} ):
      
    self.config_override = { k:ScalarNode( 'tag:yaml.org,2002:str', v, '','' )
                             for k,v in config_override.items() }
    self.unused_override = set( config_override )

    Reader.__init__(self, stream)
    Scanner.__init__(self)
    Parser.__init__(self)
    OverrideableComposer.__init__(self )
    Constructor.__init__(self)
    Resolver.__init__(self)


  @staticmethod
  def add( keyword ):
    def decorator( f ):
      CustomYamlLoader.add_constructor( keyword, f )
    return decorator


@CustomYamlLoader.add( '!join_paths' )
def yaml_join_paths( loader, node ):
  '''
  Allows joining paths in the config file, e.g.:
  "weigth_path: !join_paths [*ROOT_PATH, weights]"
  '''
  seq = loader.construct_sequence( node )
  return os.path.join( *seq )

@CustomYamlLoader.add( '!length' )
def yaml_length( loader, node ):
  '''
  Allows getting the length of some other node in the
  yaml tree
  '''
  seq = loader.construct_sequence( node )
  return len( seq[0] )

@CustomYamlLoader.add( '!nof_non_replaced_classes' )
def yaml_nof_non_replaced_classes( loader, node ):
  '''
  Really special method for getting the number of non-replaced
  classes. It's a bit dirty, so a better solution should
  probably be constructed
  yaml tree, e.g.:
  "num_outputs: !nof_non_replaced_classes [*CLASSES]"
  '''
  classes, = loader.construct_sequence( node )
  return sum( 1 for cl in classes if not 'replace_by' in cl )




def load_yaml( stream, config_override={}, strict=True ):
  loader = CustomYamlLoader( stream, config_override=config_override )
  try:
    data = loader.get_single_data()
    if strict and loader.unused_override:
      raise ValueError( 'The following config override variables \
                         where not used: %s'%loader.unused_override )
    return data
  finally:
    loader.dispose()
  