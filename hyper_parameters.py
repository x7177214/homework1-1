# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/ResNeXt-in-tensorflow
# ==============================================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'test', '''Specify that the main code is for training or testing''')

## The following flags define hyper-parameters that specifically characterize ResNeXt
tf.app.flags.DEFINE_integer('cardinality', 4, '''Cadinality, number of paths in each block''')
tf.app.flags.DEFINE_integer('block_unit_depth', 32, '''the depth(# filters) of each split. 64 for cifar10
in Figure 7 of the paper''')
tf.app.flags.DEFINE_integer('num_fc_units', 128, '''Number of neurons in the fc layer''')
tf.app.flags.DEFINE_string('bottleneck_implementation', 'b', '''To use Figure 3b or 3c to
implement''')


## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'GG123', '''A version number defining the directory to
save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 200, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_integer('save_freq', 200, '''Steps takes to save the current ckpt''')
tf.app.flags.DEFINE_integer('max_to_keep', 400, '''Max # ckpt to keep''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')

## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps', 80000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 15, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 100, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 10, '''Test batch size''')


# tf.app.flags.DEFINE_float('init_lr', 0.001, '''Initial learning rate''')
# tf.app.flags.DEFINE_float('lr_decay_factor', 0.001, '''How much to decay the learning rate each
# time''')
tf.app.flags.DEFINE_float('k', 0.5, '''k * loss_fa + (1-k) * loss_obj''')
tf.app.flags.DEFINE_float('init_lr', 0.05, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 4e-3, '''How much to decay the learning rate each
time''')

## The following flags define hyper-parameters modifying the training network
tf.app.flags.DEFINE_integer('num_resnext_blocks', 3, '''How many blocks do you want,
total layers = 3n + 2, the paper used n=3, 29 layers, as demo''')
tf.app.flags.DEFINE_float('weight_decay', 0.0007, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('ckpt_path', 'logs_oh,mfc_c=2_d=64_n=2_lr=0.1_lrd=0.0004_wd=0.0007_k=0.5/model.ckpt-39800', '''Checkpoint
directory to restore to continue TRAIN''')

# tf.app.flags.DEFINE_string('test_ckpt_path', 'logs_oh,mfc_ver2_c=4_d=32_n=3_lr=0.05_lrd=0.004_wd=0.0007_k=0.5/model.ckpt-29400', '''Checkpoint
# directory to restore to TEST''')
tf.app.flags.DEFINE_string('test_ckpt_path', 'logs_onlyhand_c=3_b=15/model.ckpt-39999', '''Checkpoint
directory to restore to TEST''')


lr_curve_file_name = 'c='+str(FLAGS.cardinality) + '_'\
    'd='+str(FLAGS.block_unit_depth) + '_'\
    'n='+str(FLAGS.num_resnext_blocks) + '_'\
    'lr='+str(FLAGS.init_lr) + '_'\
    'lrd='+str(FLAGS.lr_decay_factor) + '_'\
    'wd='+str(FLAGS.weight_decay) + '_'\
    'k='+str(FLAGS.k)
lr_curve_file_name = FLAGS.version + '_' + lr_curve_file_name
train_dir = 'logs_' + lr_curve_file_name + '/'


