import numpy as np
import tensorflow as tf

from data_set import DataSet

tf.logging.set_verbosity(tf.logging.INFO)
EMBEDDING_DIMENSION = 128
CELL_SIZE = 128
BATCH_SIZE = 256
NUM_LAYERS = 1
DROP_OUT = 0.5
LEARNING_RATE = 0.001


def input_fn(dataset: DataSet, unlabeled_dataset: DataSet = None, size: int = BATCH_SIZE):
    input_dict = {
    }

    # unlabeled data
    if unlabeled_dataset is not None:
        unlabeled_dataset = unlabeled_dataset.get_batch(size)
        input_dict['unlabeled_inputs'] = tf.constant(np.array(unlabeled_dataset.inputs()))
        input_dict['unlabeled_sequence_length'] = tf.constant(unlabeled_dataset.lengths())
        input_dict['unlabeled_mask'] = tf.constant(unlabeled_dataset.masks())
        input_dict['unlabeled_target'] = tf.constant(unlabeled_dataset.labels())
    else:
        input_dict['unlabeled_inputs'] = tf.zeros([1, DataSet.MAX_SENTENCE_LENGTH], dtype=tf.int64)
        input_dict['unlabeled_sequence_length'] = tf.zeros([1], dtype=tf.int32)
        input_dict['unlabeled_mask'] = tf.zeros([1, DataSet.MAX_SENTENCE_LENGTH], dtype=tf.float32)
        input_dict['unlabeled_target'] = tf.zeros([1, DataSet.MAX_SENTENCE_LENGTH], dtype=tf.int32)

    # labeled data
    dataset = dataset.get_batch(size)
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
    num_pos = params['num_pos']
    dropout = mode == tf.contrib.learn.ModeKeys.TRAIN and DROP_OUT or 1.0

    embeddings = tf.get_variable(
        name='embeddings',
        shape=[DataSet.vocab_size(), EMBEDDING_DIMENSION],
        initializer=tf.random_uniform_initializer(-1, 1, seed=123)
    )

    # labeled data
    labeled_inputs = features['labeled_inputs']
    labeled_length = features['labeled_sequence_length']
    labeled_mask = features['labeled_mask']
    labeled_target = target

    labeled_inputs = tf.nn.embedding_lookup(embeddings, labeled_inputs)

    # unlabeled data
    unlabeled_inputs = features['unlabeled_inputs']
    unlabeled_length = features['unlabeled_sequence_length']
    unlabeled_mask = features['unlabeled_mask']
    unlabeled_target = features['unlabeled_target']

    unlabeled_inputs = tf.nn.embedding_lookup(embeddings, unlabeled_inputs)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        inputs = tf.concat([labeled_inputs, unlabeled_inputs], 0)
        length = tf.concat([labeled_length, unlabeled_length], 0)
    else:
        inputs = labeled_inputs
        length = labeled_length

    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)

    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell,
        cell_bw=cell,
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32
    )

    activations_fw = tf.contrib.layers.fully_connected(
        inputs=outputs[0],
        num_outputs=num_classes,
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(seed=111)
    )

    activations_bw = tf.contrib.layers.fully_connected(
        inputs=outputs[1],
        num_outputs=num_classes,
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(seed=156)
    )

    labeled_activations = activations_fw + activations_bw
    labeled_prediction = tf.reshape(labeled_activations, [-1, DataSet.MAX_SENTENCE_LENGTH, num_classes])

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        labeled_prediction = tf.slice(labeled_prediction, [0, 0, 0],
                                      [BATCH_SIZE, DataSet.MAX_SENTENCE_LENGTH, num_classes])

    labeled_target = tf.one_hot(labeled_target, num_classes)
    labeled_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labeled_target,
        logits=labeled_prediction,
        weights=labeled_mask
    )

    unlabeled_loss = 0
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        activations_fw = tf.contrib.layers.fully_connected(
            inputs=outputs[0],
            num_outputs=num_pos,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(seed=211)
        )

        activations_bw = tf.contrib.layers.fully_connected(
            inputs=outputs[1],
            num_outputs=num_pos,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(seed=256)
        )

        unlabeled_activations = activations_fw + activations_bw
        unlabeled_prediction = tf.reshape(unlabeled_activations, [-1, DataSet.MAX_SENTENCE_LENGTH, num_pos])
        unlabeled_prediction = tf.slice(unlabeled_prediction, [BATCH_SIZE, 0, 0],
                                        [BATCH_SIZE, DataSet.MAX_SENTENCE_LENGTH, num_pos])

        unlabeled_target = tf.one_hot(unlabeled_target, num_pos)
        unlabeled_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=unlabeled_target,
            logits=unlabeled_prediction,
            weights=unlabeled_mask
        )

    loss = labeled_loss + unlabeled_loss * coefficient_balancing(tf.constant(300),
                                                                 tf.constant(700),
                                                                 tf.constant(3),
                                                                 tf.to_int32(tf.train.get_global_step() / 10))

    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE,
        global_step=tf.contrib.framework.get_global_step(),
        decay_steps=100,
        decay_rate=0.96
    )

    train_op = None
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer='Adam'
        )

    labeled_target = tf.argmax(labeled_target, 2)
    labeled_prediction = tf.argmax(labeled_prediction, 2)

    eval_metric_ops = None
    if mode != tf.contrib.learn.ModeKeys.INFER:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labeled_target,
                predictions=labeled_prediction,
                weights=labeled_mask
            ),
            # 'predictions': prediction,
            # 'target': target,
            # 'length': length
        }

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions={
            'predictions': labeled_prediction
        },
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def main(unused_args):
    training_set = DataSet('./atis.slot', './atis.train').sample(400)
    validation_set = DataSet('./atis.slot', './atis.dev')
    test_set = DataSet('./atis.slot', './atis.test')
    unlabeled_set = DataSet('./pos.slot', './pos.unlabeled')

    print('# training_set (%d)' % training_set.size())
    print('# validation_set (%d)' % validation_set.size())
    print('# test_set (%d)' % test_set.size())
    print('# unlabeled_set (%d)' % unlabeled_set.size())

    classifier = tf.contrib.learn.Estimator(
        model_fn=rnn_model_fn,
        params={
            'num_classes': training_set.num_classes(),
            'num_pos': unlabeled_set.num_classes(),
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
