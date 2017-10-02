# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/ResNeXt-in-tensorflow
# ==============================================================================
from cifar10_input_tfRe import *
from resNeXt import *
from datetime import datetime
import time
import pandas as pd
import skimage.io as io
import skimage.transform 
# from data_io import *

class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()


    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder_fa= tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])
        self.label_placeholder_obj = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder_fa = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])
        self.vali_label_placeholder_obj = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])



    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.

        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits_fa, logits_obj = inference(self.image_placeholder, FLAGS.num_resnext_blocks, reuse=False)
        vali_logits_fa, vali_logits_obj = inference(self.vali_image_placeholder, FLAGS.num_resnext_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss = self.loss(logits, self.label_placeholder)
        loss = self.loss(logits_fa, logits_obj, self.label_placeholder_fa, self.label_placeholder_obj, k=FLAGS.k)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions_obj = tf.nn.softmax(logits_obj)
        self.train_top1_error_obj = self.top_k_error(predictions_obj, self.label_placeholder_obj, 1)
        predictions_fa = tf.nn.softmax(logits_fa)
        self.train_top1_error_fa = self.top_k_error(predictions_fa, self.label_placeholder_fa, 1)

        # Validation loss
        # self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        self.vali_loss = self.loss(vali_logits_fa, vali_logits_obj, self.vali_label_placeholder_fa, self.vali_label_placeholder_obj, k=FLAGS.k)
        
        vali_predictions_obj = tf.nn.softmax(vali_logits_obj)
        self.vali_top1_error_obj = self.top_k_error(vali_predictions_obj, self.vali_label_placeholder_obj, 1)
        vali_predictions_fa = tf.nn.softmax(vali_logits_fa)
        self.vali_top1_error_fa = self.top_k_error(vali_predictions_fa, self.vali_label_placeholder_fa, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error_obj)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error_obj, self.vali_loss)



    def train(self):
        '''
        This is the main function for training
        '''

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print 'Restored from checkpoint...'
            starting_step = int(FLAGS.ckpt_path.split('-')[-1])
            print 'Starting step = %d' % starting_step
        else:
            sess.run(init)
            starting_step = 0

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        # These lists are used to save a csv file at last
        step_list = []
        train_error_list_fa = []
        train_error_list_obj = []
        val_error_list_fa = []
        val_error_list_obj = []

        # Prepare the validation batch data
        print 'Prepare the validation batch data...'
        print '----------------------------'
        vali_data_list = read_path_and_label('test') # load all paths to validation images into the memory
        validation_batch_hand, validation_batch_head, validation_batch_label_fa, validation_batch_label_ges, validation_batch_label_obj = \
                            self.generate_data_batch(vali_data_list, FLAGS.validation_batch_size, 'valid')

        print 'Start training...'
        print '----------------------------'

        # Define the procedure of tfRecord data -> tensor data
        tfrecords_filename = 'training_data.tfrecords'
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=None)
        # Define the procedure of getting a batch of data
        # (Even when reading in multiple threads, share the filename queue.)
        images_hand, images_head, labels_fa, labels_ges, labels_obj = self.read_and_decode(filename_queue)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in xrange(starting_step, FLAGS.train_steps):

            # Get a batch of data. (tensor data -> numpy data)
            train_batch_hand, train_batch_label_obj, train_batch_label_fa = sess.run([images_hand, labels_obj, labels_fa])       
            
            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if step % FLAGS.report_freq == 0:
                # if FLAGS.is_full_validation is True:
                #     validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                #                             top1_error=self.vali_top1_error, vali_data=vali_data,
                #                             vali_labels=vali_labels, session=sess,
                #                             batch_data=train_batch_data, batch_label=train_batch_labels)

                #     vali_summ = tf.Summary()
                #     vali_summ.value.add(tag='full_validation_error',
                #                         simple_value=validation_error_value.astype(np.float))
                #     summary_writer.add_summary(vali_summ, step)
                #     summary_writer.flush()

                # else:
                _, validation_error_value_fa, validation_error_value_obj, validation_loss_value = sess.run([self.val_op, self.vali_top1_error_fa, self.vali_top1_error_obj, self.vali_loss],
                                            {self.image_placeholder: train_batch_hand, # train_batch_data
                                                self.label_placeholder_obj: train_batch_label_obj, # train_batch_labels
                                                self.label_placeholder_fa: train_batch_label_fa, # train_batch_labels
                                                self.vali_image_placeholder: validation_batch_hand, # validation_batch_data
                                                self.vali_label_placeholder_obj: validation_batch_label_obj, # validation_batch_labels
                                                self.vali_label_placeholder_fa: validation_batch_label_fa, # validation_batch_labels
                                                self.lr_placeholder: FLAGS.init_lr})

                val_error_list_fa.append(validation_error_value_fa)
                val_error_list_obj.append(validation_error_value_obj)


            start_time = time.time()

            _, _, train_loss_value, train_error_value_fa, train_error_value_obj = sess.run([self.train_op, self.train_ema_op,
                                                           self.full_loss, self.train_top1_error_fa, self.train_top1_error_obj],
                                {self.image_placeholder: train_batch_hand, # train_batch_data
                                self.label_placeholder_obj: train_batch_label_obj, # train_batch_labels
                                self.label_placeholder_fa: train_batch_label_fa, # train_batch_labels
                                self.vali_image_placeholder: validation_batch_hand, # validation_batch_data
                                self.vali_label_placeholder_obj: validation_batch_label_obj, # validation_batch_labels
                                self.vali_label_placeholder_fa: validation_batch_label_fa, # validation_batch_labels
                                self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_hand, # train_batch_data
                                                    self.label_placeholder_obj: train_batch_label_obj, # train_batch_labels
                                                    self.label_placeholder_fa: train_batch_label_fa, # train_batch_labels
                                                    self.vali_image_placeholder: validation_batch_hand, # validation_batch_data
                                                    self.vali_label_placeholder_obj: validation_batch_label_obj, # validation_batch_labels
                                                    self.vali_label_placeholder_fa: validation_batch_label_fa, # validation_batch_labels
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch)
                print 'Train top1 error-> OBJ: %.2f'%train_error_value_obj, ', FA: %.2f'%train_error_value_fa
                print 'Validation top1 error-> OBJ: %.2f'%validation_error_value_obj, ', FA: %.2f'%validation_error_value_fa
                print 'Validation loss = ', validation_loss_value
                print '----------------------------'

                step_list.append(step)
                train_error_list_fa.append(train_error_value_fa)
                train_error_list_obj.append(train_error_value_obj)

            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print 'Learning rate decayed to ', FLAGS.init_lr

            # Save checkpoints every FLAGS.save_freq steps
            if step % FLAGS.save_freq == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step':step_list,
                                'train_error_obj':train_error_list_obj,
                                'validation_error_obj': val_error_list_obj,
                                'train_error_fa':train_error_list_fa,
                                'validation_error_fa': val_error_list_fa})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')
        coord.request_stop()
        coord.join(threads)
        return


    def test(self, test_image_array, REUSE=False):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance
        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print '%i test batches in total...' % num_batches

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_TEST_HEIGHT, IMG_TEST_WIDTH, IMG_DEPTH])

        # Build the test graph
        # logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        logits_fa, logits_obj = inference(self.test_image_placeholder, FLAGS.num_resnext_blocks, reuse=REUSE)
        predictions = tf.nn.softmax(logits_obj)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        saver.restore(sess, FLAGS.test_ckpt_path)
        print 'Model restored from ', FLAGS.test_ckpt_path

        prediction_array = np.array([]).reshape(-1, NUM_OBJ_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print '%i batches finished!' %step
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            # logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            logits_fa, logits_obj = inference(self.test_image_placeholder, FLAGS.num_resnext_blocks, reuse=False)
            predictions = tf.nn.softmax(logits_obj)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array


    ## Helper functions
    # def loss(self, logits, labels):
    #     '''
    #     Calculate the cross entropy loss given logits and true labels
    #     :param logits: 2D tensor with shape [batch_size, num_labels]
    #     :param labels: 1D tensor with shape [batch_size]
    #     :return: loss tensor with shape [1]
    #     '''
    #     labels = tf.cast(labels, tf.int64)
    #     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
    #                                                                    name='cross_entropy_per_example')
    #     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #     return cross_entropy_mean

    def loss(self, logits1, logits2, labels1, labels2, k=0.5):
        '''
        Calculate the cross entropy loss given logits and true labels (multi-task)
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :param k: weight between loss1 and loss2
        :return: loss tensor with shape [1]
        '''
        labels1 = tf.cast(labels1, tf.int64)
        labels2 = tf.cast(labels2, tf.int64)
        cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels1,
                                                                       name='cross_entropy_per_example_1')
        cross_entropy_mean1 = tf.reduce_mean(cross_entropy1, name='cross_entropy_1')
        cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels2,
                                                                       name='cross_entropy_per_example_2')
        cross_entropy_mean2 = tf.reduce_mean(cross_entropy2, name='cross_entropy_2')
        return k * cross_entropy_mean1 + (1-k) * cross_entropy_mean2


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def read_and_decode(self, filename_queue):
        '''
        [For queue loading used]
        Read and decode tfrecord data
        '''
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_hand_raw': tf.FixedLenFeature([], tf.string),
                'image_head_raw': tf.FixedLenFeature([], tf.string),
                'label_fa': tf.FixedLenFeature([], tf.int64),
                'label_ges': tf.FixedLenFeature([], tf.int64),
                'label_obj': tf.FixedLenFeature([], tf.int64)
                })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image_hand = tf.decode_raw(features['image_hand_raw'], tf.uint8)
        image_head = tf.decode_raw(features['image_head_raw'], tf.uint8)
        
        label_fa = tf.cast(features['label_fa'], tf.int32)
        label_ges = tf.cast(features['label_ges'], tf.int32)
        label_obj = tf.cast(features['label_obj'], tf.int32)
        
        Image_shape = tf.stack([IMG_TMP_HEIGHT, IMG_TMP_WIDTH, IMG_DEPTH])
        image_hand = tf.reshape(image_hand, Image_shape)
        image_head = tf.reshape(image_head, Image_shape)
        
        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        # Flip an image at 50% possibility
        image_hand = tf.image.random_flip_left_right(image_hand)
        image_head = tf.image.random_flip_left_right(image_head)

        # Linearly scales image to have zero mean and unit norm.
        image_hand = tf.image.per_image_standardization(image_hand)
        image_head = tf.image.per_image_standardization(image_head)

        resized_image_hand = tf.image.resize_images(image_hand,
                                            tf.cast([IMG_HEIGHT, IMG_WIDTH], tf.int32))
        resized_image_head = tf.image.resize_images(image_head,
                                            tf.cast([IMG_HEIGHT, IMG_WIDTH], tf.int32))

        images_hand, images_head, labels_fa, labels_ges, labels_obj = \
                                            tf.train.shuffle_batch([resized_image_hand, resized_image_head, label_fa, label_ges, label_obj],
                                                    batch_size=FLAGS.train_batch_size,
                                                    capacity=100,
                                                    num_threads=4,
                                                    min_after_dequeue=10)
        return images_hand, images_head, labels_fa, labels_ges, labels_obj

    def generate_data_batch(self, data_list, batch_size, mode, offset = 0):
        '''
        [For queue loading NOT used]
        This function helps generate a batch of train data, and horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param batch_size: int
        :param mode: string. Indicate the data_list is for train, valid or test
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        if mode is 'test':
            batch_data = data_list[offset:offset+batch_size]
        else:
            offset = np.random.choice(len(data_list) - batch_size, 1)[0] # randomly choosed offset
            batch_data = data_list[offset:offset+batch_size]

        batch_path_hand = [ele[0] for ele in batch_data]
        batch_path_head = [ele[1] for ele in batch_data]

        batch_hand_imgs = read_in_imgs(batch_path_hand, mode)
        batch_head_imgs = read_in_imgs(batch_path_head, mode)

        batch_hand_imgs = whitening_image(batch_hand_imgs)
        batch_head_imgs = whitening_image(batch_head_imgs)

        batch_label_fa = [ele[2] for ele in batch_data]
        batch_label_ges = [ele[3] for ele in batch_data]
        batch_label_obj = [ele[4] for ele in batch_data]

        return batch_hand_imgs, batch_head_imgs, batch_label_fa, batch_label_ges, batch_label_obj


    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op

def read_test_data():
    testing_img = io.imread('Image496.png')
    testing_img = skimage.transform.resize(testing_img, [IMG_HEIGHT, IMG_WIDTH], order=3, mode='reflect')
    testing_img = np.reshape(testing_img, [1, testing_img.shape[0], testing_img.shape[1], testing_img.shape[2]])
    return testing_img

# Initialize the Train object
train = Train()

print 'MODE: ' + FLAGS.mode +'ing'

if FLAGS.mode == 'test':
    '''
    # Read testing data
    NUMBER_OF_TESTING_DATA = 100
    test_data_list = read_path_and_label('test')
    print 'Prepare the testing batch data...'
    print '----------------------------'
    test_batch_hand, _, _, _, test_batch_label_obj = \
                        train.generate_data_batch(test_data_list, NUMBER_OF_TESTING_DATA, 'test')
    # Start the testing session
    prob_map = train.test(test_batch_hand)
    prediction = np.argmax(prob_map, axis=1)
    cnt = 0.0
    for pred, label in zip(prediction, test_batch_label_obj):
        if int(pred)==int(label):
            cnt = cnt + 1.0
    accuracy = float(cnt) / float(len(test_batch_label_obj))
    print accuracy
    '''

    # Read testing data
    BUFFER_SIZE = 100
    test_data_list = read_path_and_label('test')
    print 'Prepare the testing batch data...'
    print '----------------------------'

    NUMBER_OF_BUFFER = len(test_data_list) / BUFFER_SIZE
    REMINDER = len(test_data_list) % BUFFER_SIZE

    reuse = False
    cnt = 0.0
    for i in xrange(NUMBER_OF_BUFFER+1):
        if i == NUMBER_OF_BUFFER:
            if REMINDER == 0:
                break
            else:
                offset = NUMBER_OF_BUFFER * BUFFER_SIZE
                batch_size = REMINDER
        else:
            offset = i * BUFFER_SIZE
            batch_size = BUFFER_SIZE
        
        if i>0:
            reuse = True

        print i, NUMBER_OF_BUFFER

        test_batch_hand, _, _, _, test_batch_label_obj = \
                            train.generate_data_batch(test_data_list, batch_size, 'test', offset)
        # Start the testing session
        prob_map = train.test(test_batch_hand, reuse)
        prediction = np.argmax(prob_map, axis=1)
        for pred, label in zip(prediction, test_batch_label_obj):
            if int(pred)==int(label):
                cnt = cnt + 1.0

        print float(cnt) / float(BUFFER_SIZE) / float(i+1)

    accuracy = float(cnt) / float(len(test_data_list))
    print accuracy

else:
    # Start the training session
    train.train()

