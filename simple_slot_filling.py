import numpy as np
import tensorflow as tf

from data_set import DataSet

tf.logging.set_verbosity(tf.logging.INFO)
CELL_SIZE = 128
BATCH_SIZE = 64
NUM_LAYERS = 1


def input_fn(dataset: DataSet, size: int = BATCH_SIZE):
    input_dict = {
    }

    dataset = dataset.get_batch(size)
    input_dict['labeled_inputs'] = tf.constant(np.array(dataset.inputs()))
    input_dict['labeled_sequence_length'] = tf.constant(dataset.lengths())
    input_dict['labeled_mask'] = tf.constant(dataset.masks())
    labels = tf.constant(dataset.labels())

    return input_dict, labels


def rnn_model_fn(features, target, mode, params):
    num_classes = params['num_classes']
    dropout = mode == tf.contrib.learn.ModeKeys.TRAIN and 0.5 or 1.0

    # labeled data
    labeled_inputs = features['labeled_inputs']
    labeled_length = features['labeled_sequence_length']
    labeled_mask = features['labeled_mask']

    with tf.name_scope('rnn'):
        cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            inputs=labeled_inputs,
            sequence_length=labeled_length,
            dtype=tf.float32
        )

    output_fw = tf.reshape(outputs[0], [-1, CELL_SIZE])
    output_bw = tf.reshape(outputs[1], [-1, CELL_SIZE])

    with tf.name_scope('softmax'):
        weight_fw = tf.get_variable(name='weights_fw',
                                    shape=[CELL_SIZE, num_classes],
                                    initializer=tf.random_uniform_initializer(-1, 1))
        weight_bw = tf.get_variable(name='weights_bw',
                                    shape=[CELL_SIZE, num_classes],
                                    initializer=tf.random_uniform_initializer(-1, 1))
        bias = tf.get_variable(name='bias',
                               shape=[num_classes],
                               initializer=tf.random_uniform_initializer(-1, 1))
        softmax = tf.nn.softmax(tf.matmul(output_fw, weight_fw) + tf.matmul(output_bw, weight_bw) + bias)

    prediction = tf.reshape(softmax, [-1, DataSet.MAX_SENTENCE_LENGTH, num_classes])

    target = tf.one_hot(target, num_classes)
    cross_entropy = -tf.reduce_sum(target * tf.log(prediction), reduction_indices=2) * labeled_mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(labeled_length, tf.float32)
    loss = tf.reduce_mean(cross_entropy)

    train_op = None
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        learning_rate = tf.constant(0.0001)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer='RMSProp'
        )

    target = tf.argmax(target, 2)
    prediction = tf.argmax(prediction, 2)

    eval_metric_ops = None
    if mode != tf.contrib.learn.ModeKeys.INFER:

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=target,
                predictions=prediction,
                weights=labeled_mask
            ),
            # 'predictions': prediction,
            # 'target': target,
            # 'length': length
        }

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions={
            'predictions': prediction
        },
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def main(unused_argv):
    DataSet.init()
    training_set = DataSet('./atis.slot', './atis.train')
    validation_set = DataSet('./atis.slot', './atis.dev')
    test_set = DataSet('./atis.slot', './atis.test')

    print('# training_set (%d)' % training_set.size())
    print('# validation_set (%d)' % validation_set.size())
    print('# test_set (%d)' % test_set.size())

    classifier = tf.contrib.learn.Estimator(
        model_fn=rnn_model_fn,
        params={
            'num_classes': training_set.num_classes()
        },
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30),
        model_dir='./model/'
    )

    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key='predictions',
                weight_key='labeled_mask'
            )
    }

    monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: input_fn(validation_set, size=-1),
        eval_steps=1,
        every_n_steps=50,
        metrics=validation_metrics,
        # early_stopping_metric="loss",
        # early_stopping_metric_minimize=True,
        # early_stopping_rounds=500
    )

    classifier.fit(
        input_fn=lambda: input_fn(training_set),
        monitors=[monitor],
        steps=1000
    )

    accuracy_score = classifier.evaluate(
        input_fn=lambda: input_fn(test_set, size=-1),
        steps=1
    )["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))


if __name__ == "__main__":
    tf.app.run()
