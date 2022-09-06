import tensorflow as tf
import neuralgym as ng

def multigpu_graph_def(model, FLAGS, data, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(FLAGS.batch_size)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')

def static_validation(FLAGS, img_shapes, model):
    with open(FLAGS.data_flist[FLAGS.dataset][1]) as f:
            val_fnames = f.read().splitlines()
    if FLAGS.guided:
        val_fnames = [
            (fname, fname[:-4] + '_edge.jpg') for fname in val_fnames]
    # progress monitor by visualizing static images
    for i in range(FLAGS.static_view_size):
        static_fnames = val_fnames[i:i+1]
        static_images = ng.data.DataFromFNames(
            static_fnames, img_shapes, nthreads=1,
            random_crop=FLAGS.random_crop).data_pipeline(1)
        static_inpainted_images = model.build_static_infer_graph(
            FLAGS, static_images, name='static_view/%d' % i)