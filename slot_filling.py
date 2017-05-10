import tensorflow as tf

# SF_MODEL_DIR = './model_sf'
SF_MODEL_DIR = None
SF_STEPS = 3000
EMBEDDING_DIMENSION = 100
CELL_SIZE = 100
LEARNING_RATE = 0.001
MAX_SEQUENCE_LENGTH = 50
WORD_PADDING = 572
LABEL_PADDING = 127
ENTITY_PADDING = 141
TAG_PADDING = 34


class SlotFilling:

    @staticmethod
    def input_fn(data_set):
        input_dict = {
        }

        inputs = []
        named_entities = []
        pos = []
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
            tag = []
            mask = []
            for idx in range(MAX_SEQUENCE_LENGTH):
                if idx < length:
                    x.append(words[idx])
                    y.append(slots[idx])
                    entity.append(entities[idx])
                    tag.append(tags[idx])
                    mask.append(1.0)
                else:
                    x.append(WORD_PADDING)
                    y.append(LABEL_PADDING)
                    entity.append(ENTITY_PADDING)
                    tag.append(TAG_PADDING)
                    mask.append(0.0)

            inputs.append(x)
            labels.append(y)
            named_entities.append(entity)
            pos.append(tag)
            lengths.append(length)
            masks.append(mask)

        input_dict['input'] = tf.constant(inputs)
        input_dict['named_entity'] = tf.constant(named_entities)
        input_dict['pos'] = tf.constant(pos)
        input_dict['sequence_length'] = tf.constant(lengths)
        input_dict['mask'] = tf.constant(masks)
        labels = tf.constant(labels)

        return input_dict, labels

    @staticmethod
    def rnn_model_fn(features, target, mode, params):
        num_slot = params['num_slot']
        cell_size = CELL_SIZE
        embedding_dimension = EMBEDDING_DIMENSION
        vocab_size = params['vocab_size']
        entity_size = params['entity_size']
        tag_size = params['tag_size']
        learning_rate = LEARNING_RATE
        embedding_mode = params['embedding_mode']
        cnn = params['cnn']
        drop_out = mode == tf.contrib.learn.ModeKeys.TRAIN and params['drop_out'] or 1.0

        inputs = features['input']
        named_entities = features['named_entity']
        pos = features['pos']
        lengths = features['sequence_length']
        masks = features['mask']
        target = target

        def embedding(name, size, ids):
            return tf.nn.embedding_lookup(tf.get_variable(
                name=name,
                shape=[size, embedding_dimension],
                initializer=tf.random_uniform_initializer(-1, 1)
            ), ids)

        # embeddings
        inputs = embedding('word_embeddings', vocab_size, inputs)

        def conv2d(kernel_size, stride, ips):
            return tf.reshape(tf.contrib.layers.conv2d(
                inputs=ips,
                num_outputs=embedding_dimension,
                kernel_size=kernel_size,
                stride=stride,
                padding='SAME'
            ), [-1, MAX_SEQUENCE_LENGTH, embedding_dimension])

        if embedding_mode == 'ne':
            named_entities = embedding('entity_embeddings', entity_size, named_entities)

            if cnn:
                inputs = conv2d(5, 1, inputs)
                named_entities = conv2d(5, 1, named_entities)

            inputs = tf.concat([inputs, named_entities], axis=2)
        elif embedding_mode == 'pos':
            pos = embedding('pos_embeddings', tag_size, pos)

            if cnn:
                inputs = conv2d(5, 1, inputs)
                pos = conv2d(7, 1, pos)

            inputs = tf.concat([inputs, pos], axis=2)
        elif embedding_mode == 'ne_pos':
            named_entities = embedding('entity_embeddings', entity_size, named_entities)
            pos = embedding('pos_embeddings', tag_size, pos)

            if cnn:
                inputs = conv2d(5, 1, inputs)
                named_entities = conv2d(5, 1, named_entities)
                pos = conv2d(7, 1, pos)

            inputs = tf.concat([inputs, named_entities, pos], axis=2)
        else:
            if cnn:
                inputs = conv2d(5, 1, inputs)

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
    def run(cls, training_set, dev_set, test_set, num_slot, gpu_memory, random_seed, vocab_size, entity_size, tag_size,
            drop_out, embedding_mode, cnn):
        classifier = tf.contrib.learn.Estimator(
            model_fn=cls.rnn_model_fn,
            model_dir=SF_MODEL_DIR,
            config=tf.contrib.learn.RunConfig(
                gpu_memory_fraction=gpu_memory,
                tf_random_seed=random_seed,
                save_checkpoints_secs=30,
            ),
            params={
                'num_slot': num_slot + 1,
                'drop_out': drop_out,
                'vocab_size': vocab_size + 1,
                'entity_size': entity_size + 1,
                'tag_size': tag_size + 1,
                'embedding_mode': embedding_mode,
                'cnn': cnn
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
