'''
Callback for plotting confusion matrices during training over both train
and validation data; either to .png plot or to tensorboard summary.

It's a bit hacky, since keras does not like non-scalar metrics.
'''

import itertools

import matplotlib.pyplot as plt 
from matplotlib.backend.backend_agg import FigureCanvasAgg

from keras import backend as K 
from keras.callbacks import Callback

import tensorflow as tf
import numpy as np 




class ConfusionMatrix( Callback ):
    
    def __init__( self, model, classes, log_dir, to_tensorboard=True, write_every=1 ):
        super( CustomMetric, self).__init__()
        
        self.classes = classes
        self.log_dir = log_dir
        self.to_tensorboard = to_tensorboard
        self.write_every = write_every
        self.writer = tf.summary.FileWriter( self.log_dir )
        
        # Avoid adding duplicates of matrix ops since 
        # this is a hack and metrics usually are added
        # only once; at model.compile time
        if not 'custom_cmat' in model.metrics_names:
            mat_op = self.create_mat_op( model, len(classes) )
            model.metrics_names.append( 'custom_cmat' )
            model.metrics_tensors.append( mat_op )
    
            
    def on_epoch_end( self, epoch, logs=None ):
        
        if epoch%self.write_every != 0:
            return
        
        logs = logs or {}
        
        if 'custom_cmat' in logs:
            train_cmat = logs['custom_cmat']
            self.plot_cmat( train_cmat, 'train_cmat', epoch )
        
        if 'val_custom_cmat' in logs:
            valid_cmat = logs['val_custom_cmat']
            self.plot_cmat( valid_cmat, 'valid_cmat', epoch )
            
    
    def on_train_end( self, logs=None ):
        if self.writer:
            self.writer.close()
        
        
        
    def create_mat_op( self, model, n_classes ):
        
        y_true = tf.argmax( model.targets[0], axis=-1 )
        y_pred = tf.argmax( model.outputs[0], axis=-1 )
        
        with K.name_scope( 'cmat_metric' ):
            mat_op = tf.confusion_matrix( y_true, y_pred, num_classes=n_classes )
            mat_op = tf.cast( mat_op, tf.float64, name='confusion_matrix' )
            # Normalize each index to compensate for keras normalization
            mat_op = mat_op / tf.reduce_sum( mat_op )
        
        return mat_op
    
    
    def plot_cmat( self, cmat, name, epoch ):
        
        fig = mplfig.Figure( figsize=(2,2), dpi=320, facecolor='w', edgecolor='k' )
        ax  = fig.add_subplot( 1, 1, 1 )
        im  = ax.imshow( cmat, cmap='Oranges')

        # Set labels for predicted classes (bottom/vertical)
        ax.set_xlabel( 'Predicted', fontsize=7 )
        ax.set_xticks( np.arange(len( self.classes )))
        ax.set_xticklabels( self.classes, fontsize=4, rotation=-90, ha='center' )
        ax.xaxis.set_label_position( 'bottom' )
        ax.xaxis.tick_bottom()

        # Set labels for actual classes (left/horizontal)
        ax.set_ylabel( 'Actual', fontsize=7 )
        ax.set_yticks( np.arange( len( self.classes )))
        ax.set_yticklabels( self.classes, fontsize=4, va='center' )
        ax.yaxis.set_label_position( 'left' )
        ax.yaxis.tick_left()

        # Add text with class count
        sums = cmat.sum( axis=1 )
        for i, j in itertools.product( range( cmat.shape[0]), range( cmat.shape[1]) ):

            if not cmat[i,j]:
                continue
            text = '%4.1f%%'%(cmat[i,j]*100/sums[i])
            ax.text( j, i, text, horizontalalignment="center", fontsize=3, 
                                 verticalalignment='center', color="black" )
        
        
        # Make sure figure is drawn onto a canvas with correct backend
        if fig.canvas is None:
            FigureCanvasAgg( fig )
        
        # Make sure everything is drawn
        fig.tight_layout()
        
        if self.to_tensorboard:

            fig.canvas.draw()
            w,h = fig.canvas.get_width_height()

            buff = io.BytesIO()
            fig.canvas.print_png( buff )
            png_encoded = buff.getvalue()
            buff.close()

            summary_image = tf.Summary.Image( height=h, width=w, colorspace=4,
                                              encoded_image_string=png_encoded )
            summary = tf.Summary( value=[ tf.Summary.Value( tag=name, image=summary_image )])
            
            self.writer.add_summary( summary, epoch )
        
        else:
            # Otherwise, save it as png
            fig.savefig( os.path.join( self.log_dir, name ))
            



    