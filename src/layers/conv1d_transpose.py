from keras import backend as K 
import tensorflow as tf 
from keras.layers import Conv1D 
from keras.utils import conv_utils 
from keras.engine.base_layer import InputSpec




from keras.backend import tensorflow_backend as tfK
from keras.backend.common import image_data_format

# This should be something like keras.backend.conv1d_transpose in the future
def conv1d_transpose( value, filter, output_shape, stride, padding, data_format ):

  output_shape = tf.TensorShape( output_shape )
  output_shape = tf.stack([
    tf.shape(value)[i] if dim.value is None else dim
    for i, dim in enumerate( output_shape )
  ])

  return tf.contrib.nn.conv1d_transpose(
    value=value,
    filter=filter,
    output_shape=output_shape,
    stride=stride,
    padding=padding.upper(),
    data_format='NCW' if data_format == 'channel_first' else 'NWC'
  )

class Conv1DTranspose( Conv1D ):

  def __init__( self, filters, 
      kernel_size, 
      strides=1,
      padding='valid',
      output_padding=None,
      data_format=None,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs ):
      super( Conv1DTranspose, self).__init__(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs )
      # TODO: Handle `output_padding` if we ever need it
      self.output_padding = output_padding
      if self.output_padding is not None:
        self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 1, 'output_padding' )
        for stride, out_pad in zip( self.strides, self.output_padding ):
          if out_pad >= stride:
            raise ValueError( 'Stride %s must be greater than output padding %s' % (self.strides, self.output_padding))


  def build( self, input_shape ):
    # Input should be (batch, width, channels)
    if len( input_shape ) != 3:
      raise ValueError( 'Inputs should have rank 3; Received input shape:', str( input_shape ))
    # Make sure number of channels in input tensor is defined
    if self.data_format == 'channel_first':
      channel_axis = 1
    else:
      channel_axis = -1 
    if input_shape[ channel_axis ] is None:
      raise ValueError( 'The channel dimension of the inputs should be defined. Found `None`' )

    # Define kernel shape 
    input_dim = input_shape[ channel_axis ]
    kernel_shape = ( self.kernel_size[0], self.filters, input_dim )

    # Create kernel variable
    self.kernel = self.add_weight( 
      shape=kernel_shape, 
      initializer=self.kernel_initializer,
      name='kernel',
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint
    )

    # Create bias variable if not disabled
    if self.use_bias:
      self.bias = self.add_weight(
        shape=(self.filters,),
        initializer=self.bias_initializer,
        name='bias',
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint
      )
    else:
      self.bias = None

    # Set input spec.
    self.input_spec = InputSpec( ndim=3, axes={ channel_axis: input_dim })
    self.built = True

  def call( self, inputs ):
    input_shape = K.shape( inputs )
    input_shape = tuple( d.value for d in inputs.shape )
    batch_size = input_shape[0]
    if self.data_format == 'channel_first':
      w_axis = 2
    else:
      w_axis = 1

    width = input_shape[ w_axis ]
    kernel_w, = self.kernel_size
    stride_w, = self.strides
    out_pad  = self.output_padding

    # Infer dynamic output shape 
    out_width = conv_utils.deconv_length( 
      dim_size=width,
      stride_size=stride_w,
      kernel_size=kernel_w,
      padding=self.padding,
      # output_padding=out_pad # Does not exists in earlier keras versions
    )

    # Define output shape based on channel index position
    if self.data_format == 'channel_first':
      output_shape = ( batch_size, self.filters, out_width )
    else:
      output_shape = ( batch_size, out_width, self.filters )

    # output_shape = np.asarray( output_shape )

    # Apply tensorflow's conv1d transpose function
    outputs = conv1d_transpose(
      value=inputs,
      filter=self.kernel,
      output_shape=output_shape,
      stride=stride_w,
      padding=self.padding,
      data_format=self.data_format
    )

    # Add bias if enabled
    if self.use_bias:
      outputs = K.bias_add( outputs, self.bias,
                            data_format=self.data_format )

    # Apply activation function if provided
    if self.activation is not None:
      return self.activation( outputs )
    return outputs




  def compute_output_shape( self, input_shape ):
    output_shape = list( input_shape )
    if self.data_format == 'channel_first':
      c_axis, w_axis = 1, 2
    else:
      c_axis, w_axis = 2, 1

    kernel_w, = self.kernel_size
    stride_w, = self.strides 
    out_pad  = self.output_padding

    out_width = conv_utils.deconv_length( 
      dim_size=output_shape[w_axis],
      stride_size=stride_w,
      kernel_size=kernel_w,
      padding=self.padding,
      # output_padding=out_pad # Does not exists in earlier keras versions
    )
    output_shape[ c_axis ] = self.filters
    output_shape[ w_axis ] = out_width

    return tuple(output_shape) 
