import os
import glob

import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
from utils.utils import multigpu_graph_def


class Train:
    
    def __init__(self):
        self.FLAGS = ng.Config('inpaint.yml')
        pass
    
    def data_prep(self, FLAGS):
        img_shapes = FLAGS.img_shapes
        with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
            fnames = f.read().splitlines()
        if FLAGS.guided:
            fnames = [(fname, fname[:-4] + '_edge.jpg') for fname in fnames]
            img_shapes = [img_shapes, img_shapes]
            
        data = ng.data.DataFromFNames(
            fnames, img_shapes, random_crop=FLAGS.random_crop,
            nthreads=FLAGS.num_cpus_per_job)
        
        images = data.data_pipeline(FLAGS.batch_size)
        
                
        return data, images

    

        
    def create_model(self, FLAGS, data, images):
        model = InpaintCAModel()
        g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, images)
        
        ## Trainin Settings, Learning Rate, Optimizer
        lr = tf.get_variable('lr', shape=[], trainable=False, initializer=tf.constant_initializer(1e-4))
        d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        g_optimizer = d_optimizer
        
        discriminator_training_callback = ng.callbacks.SecondaryMultiGPUTrainer(
            num_gpus=FLAGS.num_gpus_per_job,
            pstep=1,
            optimizer=d_optimizer,
            var_list=d_vars,
            max_iters=1,
            grads_summary=False,
            graph_def=multigpu_graph_def,
            graph_def_kwargs={
                'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'd'},
        )
        ###########
        
        trainer = ng.train.Trainer(
                    optimizer=g_optimizer,
                        var_list=g_vars,
                        max_iters=FLAGS.max_iters,
                        
                        grads_summary=False,
                        gradient_processor=None,
                        graph_def_kwargs={
                            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'g'},
                        spe=FLAGS.train_spe,
                        log_dir=FLAGS.log_dir,
            )

        trainer.add_callbacks([
        discriminator_training_callback,
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix=FLAGS.model_restore+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(FLAGS.train_spe, trainer.context['saver'], FLAGS.log_dir+'/snap'),
        ng.callbacks.SummaryWriter((FLAGS.val_psteps//1), trainer.context['summary_writer'], tf.summary.merge_all()),
     ])
        # launch training
        trainer.train()
        
        # trainer = ng.train.MultiGPUTrainer(
        #             num_gpus=FLAGS.num_gpus_per_job,
        #             optimizer=g_optimizer,
        #             var_list=g_vars,
        #             max_iters=FLAGS.max_iters,
        #             graph_def=multigpu_graph_def,
        #             grads_summary=False,
        #             gradient_processor=None,
        #             graph_def_kwargs={
        #                 'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'g'},
        #             spe=FLAGS.train_spe,
        #             log_dir=FLAGS.log_dir,
        #         )
    
    def process(self):
        FLAGS = ng.Config('inpaint.yml')
        data, images = self.data_prep(FLAGS)
        self.create_model(FLAGS, data, images)
        
        
        
    
if __name__ == "__main__":
    obj = Train()
    obj.process(
        
    )