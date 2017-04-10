import numpy as np
import tensorflow as tf

from data_set import DataSet

tf.logging.set_verbosity(tf.logging.INFO)
CELL_SIZE = 128
BATCH_SIZE = 64
NUM_LAYERS = 1


def input_fn(dataset: DataSet, unlabeled_dataset: DataSet = None, size: int = BATCH_SIZE):
    input_dict = {
    }

    # unlabeled data
    if unlabeled_dataset is not None:
        unlabeled_dataset = unlabeled_dataset.sample(size)
        input_dict['unlabeled_inputs'] = tf.constant(np.array(unlabeled_dataset.inputs()))
        input_dict['unlabeled_sequence_length'] = tf.constant(unlabeled_dataset.lengths())
        input_dict['unlabeled_mask'] = tf.constant(unlabeled_dataset.masks())
    else:
        input_dict['unlabeled_inputs'] = tf.zeros([1, DataSet.MAX_SENTENCE_LENGTH, DataSet.EMBEDDING_DIMENSION],
                                                  dtype=tf.float32)
        input_dict['unlabeled_sequence_length'] = tf.zeros([1], dtype=tf.int32)
        input_dict['unlabeled_mask'] = tf.zeros([1, DataSet.MAX_SENTENCE_LENGTH], dtype=tf.float32)

    # labeled data
    dataset = dataset.sample(size)
    input_dict['labeled_inputs'] = tf.constant(np.array(dataset.inputs()))
    input_dict['labeled_sequence_length'] = tf.constant(dataset.lengths())
    input_dict['labeled_mask'] = tf.constant(dataset.masks())
    labels = tf.constant(dataset.labels())

    return input_dict, labels


def coefficient_balancing(t1, t2, af, t):
    if tf.less(t, t1) is not None:
        return 0
    elif tf.less_equal(t2, t) is not None:
        return af

    return (af * (t - t1)) / (t2 - t1)


def rnn_model_fn(features, target, mode, params):
    num_classes = params['num_classes']
    learning_rate = params['learning_rate']
    dropout = mode == tf.contrib.learn.ModeKeys.TRAIN and 0.5 or 1.0

    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

    # labeled data
    labeled_inputs = features['labeled_inputs']
    labeled_length = features['labeled_sequence_length']
    labeled_mask = features['labeled_mask']

    # unlabeled data
    unlabeled_inputs = features['unlabeled_inputs']
    unlabeled_length = features['unlabeled_sequence_length']
    unlabeled_mask = features['unlabeled_mask']

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        inputs = tf.concat([labeled_inputs, unlabeled_inputs], 0)
        length = tf.concat([labeled_length, unlabeled_length], 0)
    else:
        inputs = labeled_inputs
        length = labeled_length

    outputs, state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32
    )

    output = tf.reshape(outputs, [-1, CELL_SIZE])

    with tf.name_scope('softmax'):
        weight = tf.Variable(tf.truncated_normal([CELL_SIZE, num_classes], stddev=0.01), name='weights')
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='bias')
        softmax = tf.nn.softmax(tf.matmul(output, weight) + bias)

    prediction = tf.reshape(softmax, [-1, DataSet.MAX_SENTENCE_LENGTH, num_classes])

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        shape = tf.shape(prediction)
        half = tf.to_int32(shape[0] / 2)
        labeled_prediction = tf.slice(prediction, [0, 0, 0], [half, shape[1], shape[2]])
        unlabeled_prediction = tf.slice(prediction, [half, 0, 0], [half, shape[1], shape[2]])
        prediction = labeled_prediction

        target = tf.one_hot(target, num_classes)
        cross_entropy = -tf.reduce_sum(target * tf.log(labeled_prediction), reduction_indices=2) * labeled_mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(labeled_length, tf.float32)
        labeled_loss = tf.reduce_mean(cross_entropy)

        pseudo_target = tf.argmax(unlabeled_prediction, 2)
        pseudo_target = tf.one_hot(pseudo_target, num_classes)
        cross_entropy = -tf.reduce_sum(pseudo_target * tf.log(unlabeled_prediction),
                                       reduction_indices=2) * unlabeled_mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(unlabeled_length, tf.float32)
        unlabeled_loss = tf.reduce_mean(cross_entropy)

        loss = labeled_loss + unlabeled_loss * coefficient_balancing(tf.constant(300),
                                                                     tf.constant(700),
                                                                     tf.constant(3),
                                                                     tf.to_int32(tf.train.get_global_step() / 10))

    else:
        target = tf.one_hot(target, num_classes)
        cross_entropy = -tf.reduce_sum(target * tf.log(prediction), reduction_indices=2) * labeled_mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(labeled_length, tf.float32)
        loss = tf.reduce_mean(cross_entropy)

    train_op = None
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
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
    unlabeled_set = DataSet('./atis.slot', './asr.unlabeled')

    print('# training_set (%d)' % training_set.size())
    print('# validation_set (%d)' % validation_set.size())
    print('# test_set (%d)' % test_set.size())
    print('# unlabeled_set (%d)' % unlabeled_set.size())

    classifier = tf.contrib.learn.Estimator(
        model_fn=rnn_model_fn,
        params={
            'num_classes': training_set.num_classes(),
            'learning_rate': 0.01
        },
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=5),
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
        input_fn=lambda: input_fn(training_set, unlabeled_set),
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
