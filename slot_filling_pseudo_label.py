import os

import numpy as np
import tensorflow as tf

from data_set import DataSet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.logging.set_verbosity(tf.logging.INFO)
EMBEDDING_DIMENSION = 50
CELL_SIZE = 50
BATCH_SIZE = 3500
NUM_LAYERS = 1
DROP_OUT = 0.5
LEARNING_RATE = 0.001


class SlotFilling:
    @staticmethod
    def input_fn(labeled: DataSet, unlabeled: DataSet, labeled_size, unlabeled_size):
        input_dict = {
        }

        # labeled data
        labeled = labeled.get_batch(labeled_size)
        input_dict['labeled_inputs'] = tf.constant(np.array(labeled.inputs()))
        input_dict['labeled_sequence_length'] = tf.constant(labeled.lengths())
        input_dict['labeled_mask'] = tf.constant(labeled.masks())
        labels = tf.constant(labeled.labels())

        # unlabeled data
        unlabeled = unlabeled.get_batch(unlabeled_size)
        input_dict['unlabeled_inputs'] = tf.constant(np.array(unlabeled.inputs()))
        input_dict['unlabeled_sequence_length'] = tf.constant(unlabeled.lengths())
        input_dict['unlabeled_mask'] = tf.constant(unlabeled.masks())

        return input_dict, labels

    @staticmethod
    def coefficient_balancing(t1, t2, af, t):
        t1 = tf.to_float(t1)
        t2 = tf.to_float(t2)
        af = tf.to_float(af)
        t = tf.to_float(t)

        return tf.case({
            t < t1: lambda: tf.constant(0, dtype=tf.float32),
            t2 <= t: lambda: af
        }, default=lambda: (af * (t - t1)) / (t2 - t1))

    @staticmethod
    def rnn_model_fn(features, target, mode, params):
        num_slot = params['num_slot']
        embedding_dimension = params['embedding_dimension']
        vocab_size = params['vocab_size']
        drop_out = mode == tf.contrib.learn.ModeKeys.TRAIN and params['drop_out'] or 1.0

        # labeled data
        labeled_inputs = features['labeled_inputs']
        labeled_lengths = features['labeled_sequence_length']
        labeled_masks = features['labeled_mask']
        labeled_target = target

        # unlabeled data
        unlabeled_inputs = features['unlabeled_inputs']
        unlabeled_lengths = features['unlabeled_sequence_length']
        unlabeled_masks = features['unlabeled_mask']

        embeddings = tf.get_variable(
            name='embeddings',
            shape=[vocab_size, embedding_dimension],
            initializer=tf.random_uniform_initializer(-1, 1)
        )

        # embeddings
        labeled_inputs = tf.nn.embedding_lookup(embeddings, labeled_inputs)
        unlabeled_inputs = tf.nn.embedding_lookup(embeddings, unlabeled_inputs)

        inputs = tf.concat([labeled_inputs, unlabeled_inputs], axis=0)
        lengths = tf.concat([labeled_lengths, unlabeled_lengths], axis=0)

        cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=drop_out)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32
        )

        outputs_fw = outputs[0]
        outputs_bw = outputs[1]

        activations_fw = tf.contrib.layers.fully_connected(
            inputs=outputs_fw,
            num_outputs=num_slot,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='fully_connected_fw'
        )

        activations_bw = tf.contrib.layers.fully_connected(
            inputs=outputs_bw,
            num_outputs=num_slot,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='fully_connected_bw'
        )

        predictions = activations_fw + activations_bw
        labeled_predictions, unlabeled_predictions = tf.split(value=predictions,
                                                              num_or_size_splits=[tf.shape(labeled_inputs)[0],
                                                                                  tf.shape(unlabeled_inputs)[0]],
                                                              axis=0)

        labeled_target = tf.one_hot(labeled_target, num_slot)
        labeled_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labeled_target,
            logits=labeled_predictions,
            weights=labeled_masks
        )

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            unlabeled_target = tf.argmax(unlabeled_predictions, 2)
            unlabeled_target = tf.one_hot(unlabeled_target, num_slot)
            unlabeled_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=unlabeled_target,
                logits=unlabeled_predictions,
                weights=unlabeled_masks
            )

            balancing = SlotFilling.coefficient_balancing(50, 350, 3, tf.train.get_global_step())
            loss = labeled_loss + unlabeled_loss * balancing

            tf.summary.scalar('balancing', balancing)
            tf.summary.scalar('labeled_loss', labeled_loss)
            tf.summary.scalar('unlabeled_loss', unlabeled_loss)
        else:
            loss = labeled_loss

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
        labeled_predictions = tf.argmax(labeled_predictions, 2)

        eval_metric_ops = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    labels=labeled_target,
                    predictions=labeled_predictions,
                    weights=labeled_masks
                )
            }

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions={
                'predictions': labeled_predictions
            },
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

    @classmethod
    def run(cls, dev, test, labeled_slot, labeled_train, unlabeled_slot, unlabeled_train, steps, gpu_memory):
        training_set = DataSet(labeled_slot, labeled_train)
        validation_set = DataSet(labeled_slot, dev)
        test_set = DataSet(labeled_slot, test)
        unlabeled_set = DataSet(unlabeled_slot, unlabeled_train)

        print('# training_set (%d)' % training_set.size())
        print('# validation_set (%d)' % validation_set.size())
        print('# test_set (%d)' % test_set.size())
        print('# unlabeled_set (%d)' % unlabeled_set.size())

        classifier = tf.contrib.learn.Estimator(
            model_fn=SlotFilling.rnn_model_fn,
            params={
                'num_slot': training_set.num_classes(),
                'num_pos': unlabeled_set.num_classes(),
                'drop_out': DROP_OUT,
                'embedding_dimension': EMBEDDING_DIMENSION,
                'vocab_size': DataSet.vocab_size(),
                'unlabeled': unlabeled_set.size() > 0
            },
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=gpu_memory,
                save_checkpoints_secs=30,
            ),
            model_dir='./model'
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
            input_fn=lambda: SlotFilling.input_fn(validation_set, unlabeled_set, validation_set.size(), 1),
            eval_steps=1,
            every_n_steps=50,
            metrics=validation_metrics,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=300
        )

        classifier.fit(
            input_fn=lambda: SlotFilling.input_fn(training_set, unlabeled_set, training_set.size(), 500),
            monitors=[monitor],
            steps=steps
        )

        predictions = classifier.predict(
            input_fn=lambda: SlotFilling.input_fn(test_set, unlabeled_set, test_set.size(), 1)
        )

        slot_correct = 0
        slot_no_match = 0
        slot_mismatch = 0
        slot_over_match = 0

        for i, p in enumerate(predictions):
            target = test_set.labels()[i][:test_set.lengths()[i]]
            prediction = list(p['predictions'])[:test_set.lengths()[i]]
            for expected, actual in zip(target, prediction):
                actual = int(actual)
                if expected is actual:
                    slot_correct += 1
                elif test_set.get_slot(actual) is 'o':
                    slot_no_match += 1
                elif test_set.get_slot(expected) is 'o':
                    slot_over_match += 1
                else:
                    slot_mismatch += 1

        return {
            'accuracy': slot_correct / sum(test_set.lengths()),
            'correct': slot_correct,
            'no_match': slot_no_match,
            'mismatch': slot_mismatch,
            'over_match': slot_over_match,
        }


if __name__ == '__main__':
    pass
