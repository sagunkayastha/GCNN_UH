import os
import glob

import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel

from utils.callbacks import callbacks
from utils.utils import multigpu_graph_def, static_validation


class Main:
    
    def __init__(self, yml_file):
        self.FLAGS = ng.Config(yml_file)
        self.img_shapes = self.FLAGS.img_shapes
    
    def create_data(self):
        with open(self.FLAGS.data_flist[self.FLAGS.dataset][0]) as f:
            fnames = f.read().splitlines()
            
        data = ng.data.DataFromFNames(
            fnames, self.img_shapes, random_crop=self.FLAGS.random_crop,
            nthreads=self.FLAGS.num_cpus_per_job)
    
        images = data.data_pipeline(self.FLAGS.batch_size)
        
        
        return data, images
        
    def build_model(self, data, images):
        self.model = InpaintCAModel()
        g_vars, d_vars, losses = self.model.build_graph_with_losses(self.FLAGS, images)
        
        if self.FLAGS.val:
            static_inpainted_images = static_validation(self.FLAGS, self.img_shapes, self.model)
            
        
        ### Learning rate and Optimizer 
        lr = tf.get_variable('lr', shape=[], trainable=False,
                             initializer=tf.constant_initializer(1e-4))
        d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        g_optimizer = d_optimizer
        
        trainer = callbacks(self.model, self.FLAGS, g_vars, d_vars, data, losses, d_optimizer, g_optimizer)
        return trainer
        
    def run(self):
        data, images = self.create_data()
        trainer = self.build_model(data, images)
        trainer.train()
    
if __name__ == "__main__":
    obj = Main("inpaint.yml")
    obj.run()
    
