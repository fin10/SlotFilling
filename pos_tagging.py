import os

import numpy as np
import tensorflow as tf

from data_set import DataSet

POS_MODEL_DIR = './model_pos'


class PosTagging:
    @staticmethod
    def input_fn(data_set: DataSet, size):
        input_dict = {
        }

        # labeled data
        data_set = data_set.get_batch(size)
        input_dict['inputs'] = tf.constant(np.array(data_set.inputs()))
        input_dict['sequence_length'] = tf.constant(data_set.lengths())
        input_dict['mask'] = tf.constant(data_set.masks())
        labels = tf.constant(data_set.labels())

        return input_dict, labels

    @classmethod
    def rnn_model_fn(cls, features, target, mode, params):
        num_pos = params['num_pos']
        cell_size = params['cell_size']
        embedding_dimension = params['embedding_dimension']
        vocab_size = params['vocab_size']
        learning_rate = params['learning_rate']
        drop_out = mode == tf.contrib.learn.ModeKeys.TRAIN and params['drop_out'] or 1.0

        # labeled data
        inputs = features['inputs']
        lengths = features['sequence_length']
        masks = features['mask']

        with tf.variable_scope('shared'):
            embeddings = tf.get_variable(
                name='embeddings',
                shape=[vocab_size, embedding_dimension],
                initializer=tf.random_uniform_initializer(-1, 1)
            )

            # embeddings
            inputs = tf.nn.embedding_lookup(embeddings, inputs)

            cell_fw = tf.contrib.rnn.GRUCell(cell_size)
            cell_bw = tf.contrib.rnn.GRUCell(cell_size)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                sequence_length=lengths,
                dtype=tf.float32,
                scope='rnn'
            )

            activations = outputs[0] + outputs[1]

        predictions = tf.contrib.layers.fully_connected(
            inputs=activations,
            num_outputs=num_pos,
            scope='pos_nn'
        )

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(target, num_pos),
            logits=predictions,
            weights=masks
        )

        learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=tf.contrib.framework.get_global_step(),
            decay_steps=100,
            decay_rate=0.96
        )

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer='Adam'
        )

        predictions = tf.argmax(predictions, 2)

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
    def run(cls, training_set, steps, gpu_memory, random_seed, vocab_size, drop_out, cell_size, embedding_dimension,
            learning_rate):
        model_dir = '{}_{}'.format(POS_MODEL_DIR, steps)

        if os.path.exists(model_dir):
            return model_dir

        classifier = tf.contrib.learn.Estimator(
            model_fn=cls.rnn_model_fn,
            model_dir=model_dir,
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=gpu_memory,
                tf_random_seed=random_seed,
                save_checkpoints_secs=30,
            ),
            params={
                'num_pos': training_set.num_classes(),
                'drop_out': drop_out,
                'cell_size': cell_size,
                'embedding_dimension': embedding_dimension,
                'vocab_size': vocab_size,
                'learning_rate': learning_rate
            }
        )

        validation_metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key='predictions',
                    weight_key='mask'
                )
        }

        monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: cls.input_fn(training_set, training_set.size()),
            eval_steps=1,
            every_n_steps=50,
            metrics=validation_metrics,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=200
        )

        classifier.fit(
            input_fn=lambda: cls.input_fn(training_set, 3000),
            monitors=[monitor],
            steps=steps
        )

        accuracy = classifier.evaluate(
            input_fn=lambda: cls.input_fn(training_set, training_set.size()),
            steps=1
        )['accuracy']

        print('# Accuracy: {0:f}'.format(accuracy))

        return model_dir


if __name__ == '__main__':
    pass
