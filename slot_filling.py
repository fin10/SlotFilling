import tensorflow as tf

# SF_MODEL_DIR = './model_sf'
SF_MODEL_DIR = None
SF_STEPS = 3000
MAX_SEQUENCE_LENGTH = 50
PADDING = 999


class SlotFilling:

    @staticmethod
    def input_fn(data_set):
        input_dict = {
        }

        inputs = []
        named_entities = []
        lengths = []
        masks = []
        labels = []
        for words, entities, slots in zip(data_set[0], data_set[1], data_set[2]):
            length = len(words)
            if length > MAX_SEQUENCE_LENGTH:
                raise ValueError('Length should be lower than %d: %d' % (MAX_SEQUENCE_LENGTH, length))

            x = []
            y = []
            entity = []
            mask = []
            for idx in range(MAX_SEQUENCE_LENGTH):
                x.append(idx < length and words[idx] or PADDING)
                y.append(idx < length and slots[idx] or PADDING)
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

    @staticmethod
    def rnn_model_fn(features, target, mode, params):
        num_slot = params['num_slot']
        cell_size = params['cell_size']
        embedding_dimension = params['embedding_dimension']
        vocab_size = params['vocab_size']
        entity_size = params['entity_size']
        learning_rate = params['learning_rate']
        pos_model_dir = params['pos_model_dir']
        drop_out = mode == tf.contrib.learn.ModeKeys.TRAIN and params['drop_out'] or 1.0

        inputs = features['input']
        named_entities = features['named_entity']
        lengths = features['sequence_length']
        masks = features['mask']
        target = target

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

        if pos_model_dir is not None:
            tf.contrib.framework.init_from_checkpoint(pos_model_dir, {
                'shared/': 'shared/'
            })

        predictions = tf.contrib.layers.fully_connected(
            inputs=activations,
            num_outputs=num_slot,
            scope='slot_nn'
        )

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(target, num_slot),
            logits=predictions,
            weights=masks
        )

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

        predictions = tf.argmax(predictions, 2)

        eval_metric_ops = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            tps = []
            fps = []
            fns = []
            for i in range(num_slot):
                index_map = tf.one_hot(i, depth=num_slot)
                _, tp = tf.contrib.metrics.streaming_true_positives(
                    labels=tf.gather(index_map, target),
                    predictions=tf.gather(index_map, predictions),
                    weights=masks
                )

                _, fp = tf.contrib.metrics.streaming_false_positives(
                    labels=tf.gather(index_map, target),
                    predictions=tf.gather(index_map, predictions),
                    weights=masks
                )

                _, fn = tf.contrib.metrics.streaming_false_negatives(
                    labels=tf.gather(index_map, target),
                    predictions=tf.gather(index_map, predictions),
                    weights=masks
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
                    labels=target,
                    predictions=predictions,
                    weights=masks
                ),
                'precision': precision,
                'recall': recall,
                'f-measure': 2 * (precision * recall) / (precision + recall)
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
    def run(cls, training_set, dev_set, test_set, num_slot, gpu_memory, random_seed, vocab_size, entity_size, drop_out,
            cell_size,
            embedding_dimension, learning_rate, pos_model_dir):
        classifier = tf.contrib.learn.Estimator(
            model_fn=cls.rnn_model_fn,
            model_dir=SF_MODEL_DIR,
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=gpu_memory,
                tf_random_seed=random_seed,
                save_checkpoints_secs=30,
            ),
            params={
                'num_slot': num_slot,
                'drop_out': drop_out,
                'cell_size': cell_size,
                'embedding_dimension': embedding_dimension,
                'vocab_size': vocab_size,
                'entity_size': entity_size,
                'learning_rate': learning_rate,
                'pos_model_dir': pos_model_dir,
            },
        )

        validation_metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key='predictions',
                weight_key='mask'
            ),
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
            steps=SF_STEPS
        )

        ev = classifier.evaluate(
            input_fn=lambda: cls.input_fn(test_set),
            steps=1
        )

        return {
            'accuracy': ev['accuracy'],
            'f_measure': ev['f-measure'],
            'precision': ev['precision'],
            'recall': ev['recall']
        }

if __name__ == '__main__':
    pass
