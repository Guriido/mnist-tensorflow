import tensorflow as tf
import os


LEARNING_RATE = 1e-4


# model definition
class SimpleCNN(tf.keras.models.Model):

    def __init__(self, kernel_size=5, filter_sizes=(20, 50), data_format=None):
        super(SimpleCNN, self).__init__()
        filters1, filters2 = filter_sizes

        if data_format is None or data_format == 'channels_last':
            self.c_axis = -1
        else:
            self.c_axis = 1

        self.conv2a = tf.keras.layers.Conv2D(filters1, kernel_size=kernel_size, strides=1, padding='same',
                                             data_format=data_format)
        self.bn2a = tf.keras.layers.BatchNormalization(axis=self.c_axis)
        self.mp1 = tf.keras.layers.MaxPool2D((2, 2), data_format=data_format)

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size=kernel_size, padding='same', data_format=data_format)
        self.bn2b = tf.keras.layers.BatchNormalization(axis=self.c_axis)
        self.mp2 = tf.keras.layers.MaxPool2D((2, 2), data_format=data_format)

        self.flat = tf.keras.layers.Flatten()

        self.linear1 = tf.keras.layers.Dense(500)
        self.linear2 = tf.keras.layers.Dense(10)

    def call(self, input_tensor, training=False, mask=None):
        h = tf.expand_dims(input_tensor, axis=self.c_axis)
        h = self.conv2a(h)
        h = self.bn2a(h, training=training)
        h = tf.nn.relu(h)
        h = self.mp1(h)
        h = self.conv2b(h)
        h = self.bn2b(h, training=training)
        h = tf.nn.relu(h)
        h = self.mp2(h)

        h = self.flat(h)
        h = self.linear1(h)
        h = self.linear2(h)

        return h


# dataset
def get_datasets(batchsize=128):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train.astype('float32'), y_train.astype('int32')))
    train_dataset = train_dataset.shuffle(50000).batch(batchsize)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test.astype('float32'), y_test.astype('int32')))
    test_dataset = test_dataset.batch(batchsize)

    return train_dataset, test_dataset


def build_test_graph(model):
    x = tf.random_normal((1, 28, 28, 3))
    res = model(x)
    return res


def test_call():
    with tf.Session() as sess:
        model = SimpleCNN()
        # zeros = tf.constant(0, dtype='float32', shape=(1, 28, 28, 3))
        test_resultat = build_test_graph(model)

        init = tf.global_variables_initializer()
        sess.run(init)

        print(sess.run(test_resultat))

        sess.close()

    # sess.run(model(zeros))

    # print(sess.run(model.summary()))


def past_stop_threshold(stop_threshold, eval_metric):
    """Return a boolean representing whether a model should be stopped.

    Args:
      stop_threshold: float, the threshold above which a model should stop
        training.
      eval_metric: float, the current value of the relevant metric to check.

    Returns:
      True if training should stop, False otherwise.

    Raises:
      ValueError: if either stop_threshold or eval_metric is not a number
    """
    if stop_threshold is None:
        return False

    # if not isinstance(stop_threshold, numbers.Number):
    #     raise ValueError("Threshold for checking stop conditions must be a number.")
    # if not isinstance(eval_metric, numbers.Number):
    #     raise ValueError("Eval metric being checked against stop conditions "
    #                      "must be a number.")

    if eval_metric >= stop_threshold:
        tf.logging.info(
            "Stop threshold of {} was passed with metric value {}.".format(
                stop_threshold, eval_metric))
        return True

    return False


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    # model = create_model(params['data_format'])
    model = SimpleCNN(data_format=params['data_format'])
    image = features
    if isinstance(image, dict):
        image = features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':
                    tf.metrics.accuracy(
                        labels=labels, predictions=tf.argmax(logits, axis=1)),
            })


def run_mnist(parameter_dict):
    """Run MNIST training and eval loop.

    Args:
      parameter_dict: A dictionnary containing parameters for training
    """

    model_function = model_fn

    # if flags_obj.multi_gpu:
    #     validate_batch_size_for_multi_gpu(flags_obj.batch_size)
    #
    #     # There are two steps required if using multi-GPU: (1) wrap the model_fn,
    #     # and (2) wrap the optimizer. The first happens here, and (2) happens
    #     # in the model_fn itself when the optimizer is defined.
    #     model_function = tf.contrib.estimator.replicate_model_fn(
    #         model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    # model_dir = '/tmp/mnist_model'

    data_format = None
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=parameter_dict['export_dir'],
        params={
            'data_format': data_format,
            'multi_gpu': False
        })

    batchsize = 128
    epochs_between_evals = 1
    # train, test = get_datasets(batchsize=batchsize)

    def train_input_fn():
        train, _ = get_datasets(batchsize=batchsize)
        ds = train.cache().repeat(epochs_between_evals)
        return ds

    def eval_input_fn():
        _, test = get_datasets(batchsize=batchsize)
        return test.make_one_shot_iterator().get_next()

    # # Set up training and evaluation input functions.
    # def train_input_fn():
    #     """Prepare data for training."""
    #
    #     # When choosing shuffle buffer sizes, larger sizes result in better
    #     # randomness, while smaller sizes use less memory. MNIST is a small
    #     # enough dataset that we can easily shuffle the full epoch.
    #     ds = dataset.train(flags_obj.data_dir)
    #     ds =
    #     ds = ds.cache().shuffle(buffer_size=50000).batch(flags_obj.batch_size)
    #
    #     # Iterate through the dataset a set number (`epochs_between_evals`) of times
    #     # during each training session.
    #     ds = ds.repeat(flags_obj.epochs_between_evals)
    #     return ds
    #
    # def eval_input_fn():
    #     return dataset.test(flags_obj.data_dir).batch(
    #         flags_obj.batch_size).make_one_shot_iterator().get_next()

    # Set up hook that outputs training logs every 100 steps.
    _TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                            'cross_entropy',
                                            'train_accuracy'])

    hook = tf.train.LoggingTensorHook(tensors=_TENSORS_TO_LOG,
                                      every_n_iter=100)

    train_hooks = [hook]
    # train_hooks = hooks_helper.get_train_hooks(
    #     flags_obj.hooks, batch_size=flags_obj.batch_size)

    # Train and evaluate model.
    for _ in range(parameter_dict['train_epochs'] // parameter_dict['epochs_between_evals']):
        mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print('\nEvaluation results:\n\t%s\n' % eval_results)

        if past_stop_threshold(stop_threshold=None,
                               eval_metric=eval_results['accuracy']):
            break

    # Export the model
    if parameter_dict['export_dir'] is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        mnist_classifier.export_savedmodel(parameter_dict['export_dir'], input_fn)


# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     define_mnist_flags()
#     absl_app.run(main)

if __name__ == "__main__":
    # test_call()

    options_dict = {
        'data_dir': '/tmp/mnist_data',
        'model_dir': '/tmp/mnist_model',
        'batch_size': 100,
        'train_epochs': 10,
        'epochs_between_evals': 1,
        'multi_gpu': False,
        'num_gpu': False,
        'inter_op_parallelism_threads':  0,
        'intra_op_parallelism_threads': 0,
        'export_dir': 'result',
        'stop_threshold': None,
    }

    # tf.logging.set_verbosity(tf.logging.INFO)
    # define_mnist_flags()
    # absl_app.run(main)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run_mnist(options_dict)
