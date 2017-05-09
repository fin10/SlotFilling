import os

import tensorflow as tf

POS_MODEL_DIR = './model_pos'
POS_STEPS = 3000
MAX_SEQUENCE_LENGTH = 50
PADDING = 999


class PosTagging:

    @staticmethod
    def input_fn(data_set):
        input_dict = {
        }

        inputs = []
        named_entities = []
        lengths = []
        masks = []
        labels = []
        for words, entities, slots, tags in zip(data_set[0], data_set[1], data_set[2], data_set[3]):
            length = len(words)
            if length > MAX_SEQUENCE_LENGTH:
                raise ValueError('Length should be lower than %d: %d' % (MAX_SEQUENCE_LENGTH, length))

            x = []
            y = []
            entity = []
            mask = []
            for idx in range(MAX_SEQUENCE_LENGTH):
                x.append(idx < length and words[idx] or PADDING)
                y.append(idx < length and tags[idx] or PADDING)
                entity.append(idx < length and entities[idx] or PADDING)
                mask.append(idx < length and 1 or 0)

            inputs.append(x)
            named_entities.append(entity)
            labels.append(y)
            lengths.append(length)
            masks.append(mask)

        input_dict['input'] = tf.constant(inputs)
        input_dict['named_entity'] = tf.constant(named_entities)
        input_dict['sequence_length'] = tf.constant(lengths)
        input_dict['mask'] = tf.constant(masks)
        labels = tf.constant(labels)

        return input_dict, labels

    @classmethod
    def rnn_model_fn(cls, features, target, mode, params):
        num_pos = params['num_pos']
        cell_size = params['cell_size']
        embedding_dimension = params['embedding_dimension']
        vocab_size = params['vocab_size']
        entity_size = params['entity_size']
        learning_rate = params['learning_rate']
        drop_out = mode == tf.contrib.learn.ModeKeys.TRAIN and params['drop_out'] or 1.0

        inputs = features['input']
        named_entities = features['named_entity']
        lengths = features['sequence_length']
        masks = features['mask']

        with tf.variable_scope('shared'):
            word_embeddings = tf.get_variable(
                name='word_embeddings',
                shape=[vocab_size, embedding_dimension],
                initializer=tf.random_uniform_initializer(-1, 1)
            )

            entity_embeddings = tf.get_variable(
                name='entity_embeddings',
                shape=[entity_size, embedding_dimension],
                initializer=tf.random_uniform_initializer(-1, 1)
            )

            # embeddings
            inputs = tf.nn.embedding_lookup(word_embeddings, inputs)
            named_entities = tf.nn.embedding_lookup(entity_embeddings, named_entities)

            cell_fw = tf.contrib.rnn.GRUCell(cell_size)
            cell_bw = tf.contrib.rnn.GRUCell(cell_size)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=tf.concat([inputs, named_entities], axis=2),
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
    def run(cls, training_set, dev_set, test_set, num_pos, gpu_memory, random_seed, vocab_size, entity_size, drop_out,
            cell_size, embedding_dimension,
            learning_rate):
        if os.path.exists(POS_MODEL_DIR):
            return POS_MODEL_DIR

        classifier = tf.contrib.learn.Estimator(
            model_fn=cls.rnn_model_fn,
            model_dir=POS_MODEL_DIR,
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=gpu_memory,
                tf_random_seed=random_seed,
                save_checkpoints_secs=30,
            ),
            params={
                'num_pos': num_pos,
                'drop_out': drop_out,
                'cell_size': cell_size,
                'embedding_dimension': embedding_dimension,
                'vocab_size': vocab_size,
                'entity_size': entity_size,
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
            input_fn=lambda: cls.input_fn(dev_set),
            eval_steps=1,
            every_n_steps=50,
            metrics=validation_metrics,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=100
        )

        classifier.fit(
            input_fn=lambda: cls.input_fn(training_set),
            monitors=[monitor],
            steps=POS_STEPS
        )

        accuracy = classifier.evaluate(
            input_fn=lambda: cls.input_fn(test_set),
            steps=1
        )['accuracy']

        print('# Accuracy: {0:f}'.format(accuracy))

        return POS_MODEL_DIR


if __name__ == '__main__':
    pass
