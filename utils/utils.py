import tensorflow as tf

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