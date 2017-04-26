import os

import numpy as np
import tensorflow as tf

from data_set import DataSet

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.logging.set_verbosity(tf.logging.INFO)
EMBEDDING_DIMENSION = 50
CELL_SIZE = 50
BATCH_SIZE = 3500
NUM_LAYERS = 1
DROP_OUT = 0.5
LEARNING_RATE = 0.001
CLASSIFICATION = ['pos', 'slot_filling'][0]


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
        input_dict['unlabeled_labels'] = tf.constant(unlabeled.labels())

        return input_dict, labels

    @staticmethod
    def rnn_model_fn(features, target, mode, params):
        num_slot = params['num_slot']
        num_pos = params['num_pos']
        embedding_dimension = params['embedding_dimension']
        vocab_size = params['vocab_size']
        classification = params['classification']
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
        unlabeled_target = features['unlabeled_labels']

        embeddings = tf.get_variable(
            name='embeddings',
            shape=[vocab_size, embedding_dimension],
            initializer=tf.random_uniform_initializer(-1, 1)
        )

        # embeddings
        labeled_inputs = tf.nn.embedding_lookup(embeddings, labeled_inputs)
        unlabeled_inputs = tf.nn.embedding_lookup(embeddings, unlabeled_inputs)

        if mode != tf.contrib.learn.ModeKeys.TRAIN or classification == 'pos':
            inputs = unlabeled_inputs
            lengths = unlabeled_lengths
        else:
            inputs = labeled_inputs
            lengths = labeled_lengths

        cell_fw = tf.contrib.rnn.GRUCell(CELL_SIZE)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=drop_out)

        cell_bw = tf.contrib.rnn.GRUCell(CELL_SIZE)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=drop_out)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32
        )

        outputs_fw = outputs[0]
        outputs_bw = outputs[1]

        activations_fw = tf.contrib.layers.fully_connected(
            inputs=outputs_fw,
            num_outputs=CELL_SIZE,
            scope='fully_connected_fw'
        )

        activations_bw = tf.contrib.layers.fully_connected(
            inputs=outputs_bw,
            num_outputs=CELL_SIZE,
            scope='fully_connected_bw'
        )

        activations = activations_fw + activations_bw

        if mode != tf.contrib.learn.ModeKeys.TRAIN or classification == 'pos':
            unlabeled_predictions = tf.contrib.layers.fully_connected(
                inputs=activations,
                num_outputs=num_pos,
                scope='fully_connected_pos'
            )

            unlabeled_target = tf.one_hot(unlabeled_target, num_pos)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=unlabeled_target,
                logits=unlabeled_predictions,
                weights=unlabeled_masks
            )

            predictions = tf.argmax(unlabeled_predictions, 2)
            target = tf.argmax(unlabeled_target, 2)
            masks = unlabeled_masks

        else:
            labeled_predictions = tf.contrib.layers.fully_connected(
                inputs=activations,
                num_outputs=num_slot,
                scope='fully_connected_slot'
            )

            labeled_target = tf.one_hot(labeled_target, num_slot)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labeled_target,
                logits=labeled_predictions,
                weights=labeled_masks
            )

            predictions = tf.argmax(labeled_predictions, 2)
            target = tf.argmax(labeled_target, 2)
            masks = labeled_masks

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

        eval_metric_ops = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    labels=target,
                    predictions=predictions,
                    weights=masks
                )
            }

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions={
                'predictions': predictions
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
                'classification': CLASSIFICATION
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
            # early_stopping_metric="loss",
            # early_stopping_metric_minimize=True,
            # early_stopping_rounds=300
        )

        classifier.fit(
            input_fn=lambda: SlotFilling.input_fn(training_set, unlabeled_set, training_set.size(), 1),
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
