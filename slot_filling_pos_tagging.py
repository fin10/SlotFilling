import numpy as np
import tensorflow as tf

from data_set import DataSet

# SF_MODEL_DIR = './model_sf'
SF_MODEL_DIR = None
STEPS = 3000


class SlotFilling:

    @staticmethod
    def input_fn(labeled: DataSet, labeled_size: int, unlabeled: DataSet, unlabeled_size: int):
        input_dict = {
        }

        # labeled data
        labeled = labeled.get_batch(labeled_size)
        input_dict['labeled_inputs'] = tf.constant(np.array(labeled.inputs()))
        input_dict['labeled_sequence_length'] = tf.constant(labeled.lengths())
        input_dict['labeled_mask'] = tf.constant(labeled.masks())
        input_dict['labeled_size'] = tf.constant(labeled.size())
        labels = tf.constant(labeled.labels())

        # unlabeled data
        unlabeled = unlabeled.get_batch(unlabeled_size)
        input_dict['unlabeled_inputs'] = tf.constant(np.array(unlabeled.inputs()))
        input_dict['unlabeled_sequence_length'] = tf.constant(unlabeled.lengths())
        input_dict['unlabeled_mask'] = tf.constant(unlabeled.masks())
        input_dict['unlabeled_size'] = tf.constant(unlabeled.size())

        return input_dict, labels

    @staticmethod
    def rnn_model_fn(features, target, mode, params):
        num_slot = params['num_slot']
        cell_size = params['cell_size']
        embedding_dimension = params['embedding_dimension']
        vocab_size = params['vocab_size']
        learning_rate = params['learning_rate']
        pos_model_dir = params['pos_model_dir']
        start, end, value = params['pseudo_params']
        drop_out = mode == tf.contrib.learn.ModeKeys.TRAIN and params['drop_out'] or 1.0

        # labeled data
        labeled_inputs = features['labeled_inputs']
        labeled_lengths = features['labeled_sequence_length']
        labeled_masks = features['labeled_mask']
        labeled_size = features['labeled_size']
        labeled_target = target

        # unlabeled data
        unlabeled_inputs = features['unlabeled_inputs']
        unlabeled_lengths = features['unlabeled_sequence_length']
        unlabeled_masks = features['unlabeled_mask']
        unlabeled_size = features['unlabeled_size']

        with tf.variable_scope('shared'):
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
            # inputs = labeled_inputs
            # lengths = labeled_lengths

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

        if pos_model_dir is not None:
            tf.contrib.framework.init_from_checkpoint(pos_model_dir, {
                'shared/': 'shared/'
            })

        predictions = tf.contrib.layers.fully_connected(
            inputs=activations,
            num_outputs=num_slot,
            scope='slot_nn'
        )

        labeled_predictions, unlabeled_predictions = tf.split(predictions, [labeled_size, unlabeled_size])
        # labeled_predictions = predictions

        labeled_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(labeled_target, num_slot),
            logits=labeled_predictions,
            weights=labeled_masks
        )

        unlabeled_loss = 0
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            unlabeled_target = tf.argmax(unlabeled_predictions, 2)
            unlabeled_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(unlabeled_target, num_slot),
                logits=unlabeled_predictions,
                weights=unlabeled_masks
            )

        def balancing(t1, t2, af, t):
            t1 = tf.to_float(t1)
            t2 = tf.to_float(t2)
            af = tf.to_float(af)
            t = tf.to_float(t)

            return tf.case({
                t < t1: lambda: tf.constant(0, dtype=tf.float32),
                t2 <= t: lambda: af
            }, default=lambda: (af * (t - t1)) / (t2 - t1))

        loss = labeled_loss + unlabeled_loss * balancing(start, end, value, tf.contrib.framework.get_global_step())

        learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
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

        labeled_predictions = tf.argmax(labeled_predictions, 2)

        eval_metric_ops = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            tps = []
            fps = []
            fns = []
            for i in range(num_slot):
                index_map = tf.one_hot(i, depth=num_slot)
                _, tp = tf.contrib.metrics.streaming_true_positives(
                    labels=tf.gather(index_map, labeled_target),
                    predictions=tf.gather(index_map, labeled_predictions),
                    weights=labeled_masks
                )

                _, fp = tf.contrib.metrics.streaming_false_positives(
                    labels=tf.gather(index_map, labeled_target),
                    predictions=tf.gather(index_map, labeled_predictions),
                    weights=labeled_masks
                )

                _, fn = tf.contrib.metrics.streaming_false_negatives(
                    labels=tf.gather(index_map, labeled_target),
                    predictions=tf.gather(index_map, labeled_predictions),
                    weights=labeled_masks
                )

                tps.append(tp)
                fps.append(fp)
                fns.append(fn)

            tp = sum(tps)
            fp = sum(fps)
            fn = sum(fns)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            eval_metric_ops = {
                'accuracy': tf.contrib.metrics.streaming_accuracy(
                    labels=labeled_target,
                    predictions=labeled_predictions,
                    weights=labeled_masks
                ),
                'precision': precision,
                'recall': recall,
                'f-measure': 2 * (precision * recall) / (precision + recall)
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
    def run(cls, training_set, dev_set, test_set, pseudo_set, gpu_memory, random_seed, vocab_size, drop_out, cell_size,
            embedding_dimension, learning_rate, pos_model_dir, pseudo_params):
        classifier = tf.contrib.learn.Estimator(
            model_fn=cls.rnn_model_fn,
            model_dir=SF_MODEL_DIR,
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=gpu_memory,
                tf_random_seed=random_seed,
                save_checkpoints_secs=30,
            ),
            params={
                'num_slot': training_set.num_classes(),
                'drop_out': drop_out,
                'cell_size': cell_size,
                'embedding_dimension': embedding_dimension,
                'vocab_size': vocab_size,
                'learning_rate': learning_rate,
                'pos_model_dir': pos_model_dir,
                'pseudo_params': pseudo_params
            },
        )

        validation_metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key='predictions',
                weight_key='labeled_mask'
            ),
        }

        monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: cls.input_fn(dev_set, dev_set.size(), pseudo_set, 1),
            eval_steps=1,
            every_n_steps=50,
            metrics=validation_metrics,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=100
        )

        classifier.fit(
            input_fn=lambda: cls.input_fn(training_set, training_set.size(), pseudo_set, 1),
            monitors=[monitor],
            steps=STEPS
        )

        ev = classifier.evaluate(
            input_fn=lambda: cls.input_fn(test_set, test_set.size(), pseudo_set, 1),
            steps=1
        )

        accuracy = ev['accuracy']
        precision = ev['precision']
        recall = ev['recall']
        f_measure = ev['f-measure']

        predictions = classifier.predict(
            input_fn=lambda: SlotFilling.input_fn(test_set, test_set.size(), pseudo_set, 1),
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
            'ev_accuracy': accuracy,
            'ev_precision': precision,
            'ev_recall': recall,
            'ev_f_measure': f_measure,
            'accuracy': slot_correct / sum(test_set.lengths()),
            'correct': slot_correct,
            'no_match': slot_no_match,
            'mismatch': slot_mismatch,
            'over_match': slot_over_match,
        }


if __name__ == '__main__':
    pass
