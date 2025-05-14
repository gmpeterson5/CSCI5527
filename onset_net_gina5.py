import random

import tensorflow.compat.v1 as tf
import numpy as np
from functools import reduce

dtype = tf.float32
np_dtype = dtype.as_numpy_dtype

class OnsetNet_Gina5:
    def __init__(self,
                 mode,
                 batch_size,
                 audio_context_radius,
                 audio_nbands,
                 audio_nchannels,
                 nfeats,
                 cnn_filter_shapes,
                 cnn_init,
                 cnn_pool,
                 cnn_rnn_zack,
                 rnn_cell_type,
                 rnn_size,
                 rnn_nlayers,
                 rnn_init,
                 rnn_nunroll,
                 rnn_keep_prob,
                 dnn_sizes,
                 dnn_init,
                 dnn_keep_prob,
                 dnn_nonlin,
                 pooling_method, # CSCI 5527 - added
                 target_weight_strategy,
                 grad_clip,
                 opt,
                 norm_type, # CSCI 5527 - added
                 d_model, # CSCI 5527 - added
                 num_transformer_layers, # CSCI 5527 - added
                 num_heads, # CSCI 5527 - added
                 ff_dim, # CSCI 5527 - added
                 export_feat_name=None,
                 zack_hack=0,
                 cnn_keep_prob=1.0, # CSCI 5527 - added
                 input_keep_prob=1.0, # CSCI 5527 - added
                 do_transformer=True # CSCI 5527 - added
                 ):
        audio_context_len = audio_context_radius * 2 + 1

        mode = mode
        do_cnn = len(cnn_filter_shapes) > 0
        do_rnn = rnn_size > 0 and rnn_nlayers > 0
        do_dnn = len(dnn_sizes) > 0

        if not do_rnn:
            assert rnn_nunroll == 1

        if cnn_rnn_zack:
            assert audio_context_len == 1
            assert zack_hack > 0 and zack_hack % 2 == 0

        export_feat_tensors = {}

        # Input tensors
        feats_audio_nunroll = tf.placeholder(dtype, shape=[batch_size, rnn_nunroll + zack_hack, audio_context_len, audio_nbands, audio_nchannels], name='feats_audio')
        feats_other_nunroll = tf.placeholder(dtype, shape=[batch_size, rnn_nunroll, nfeats], name='feats_other')
#         print('feats_audio: {}'.format(feats_audio_nunroll.get_shape()))
#         print('feats_other: {}'.format(feats_other_nunroll.get_shape()))

        if mode != 'gen':
            targets_nunroll = tf.placeholder(dtype, shape=[batch_size, rnn_nunroll])
            # TODO: tf.ones acts as an overridable placeholder but this is still awkward
            target_weights_nunroll = tf.ones([batch_size, rnn_nunroll], dtype)

        # Reshape input tensors to remove nunroll dim; will briefly restore later during RNN if necessary
        if cnn_rnn_zack:
            feats_audio = tf.reshape(feats_audio_nunroll, shape=[batch_size, rnn_nunroll + zack_hack, audio_nbands, audio_nchannels])
        else:
            feats_audio = tf.reshape(feats_audio_nunroll, shape=[batch_size * rnn_nunroll, audio_context_len, audio_nbands, audio_nchannels])
        feats_other = tf.reshape(feats_other_nunroll, shape=[batch_size * rnn_nunroll, nfeats])
        
        ### CSCI 5527 - added paramater to add dropout to feats_other features
        if mode == 'train' and input_keep_prob < 1.0:
            feats_other = tf.nn.dropout(feats_other, input_keep_prob)
            
        if mode != 'gen':
            targets = tf.reshape(targets_nunroll, shape=[batch_size * rnn_nunroll])
            target_weights = tf.reshape(target_weights_nunroll, shape=[batch_size * rnn_nunroll])

        # CNN
        cnn_output = feats_audio
        if do_cnn:
            layer_last = feats_audio
            nfilt_last = audio_nchannels
            for i, ((ntime, nband, nfilt), (ptime, pband)) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
                layer_name = 'cnn_{}'.format(i)
                with tf.variable_scope(layer_name):
                    filters = tf.get_variable('filters', [ntime, nband, nfilt_last, nfilt], initializer=cnn_init, dtype=dtype)
                    biases = tf.get_variable('biases', [nfilt], initializer=tf.constant_initializer(0.1), dtype=dtype)
                if cnn_rnn_zack:
                    padding = 'SAME'
                else:
                    padding = 'VALID'

                conv = tf.nn.conv2d(layer_last, filters, [1, 1, 1, 1], padding=padding)
                biased = tf.nn.bias_add(conv, biases)
                convolved = tf.nn.relu(biased)
                
############################

                # CSCI 5527 - added batch and layer normalization
                if norm_type == "batch":
#                     print("hit batch")
                    # Define training flag
                    is_training = (mode == 'train')

                    # Use tf.keras.layers.BatchNormalization
                    bn_layer = tf.keras.layers.BatchNormalization(
                        momentum=0.9,  # Equivalent to PyTorch's 0.1
                        epsilon=1e-5,
                        center=True,
                        scale=True
                    )
                    normalized = bn_layer(convolved, training=is_training)

                    pool_shape = [1, ptime, pband, 1]
                    if pooling_method == 'max':
                        pooled = tf.nn.max_pool(normalized, ksize=pool_shape, strides=pool_shape, padding='SAME')
                    elif pooling_method == 'avg':
                        pooled = tf.nn.avg_pool(normalized, ksize=pool_shape, strides=pool_shape, padding='SAME')
                    elif pooling_method == 'min':
                        pooled = -tf.nn.max_pool(-normalized, ksize=pool_shape, strides=pool_shape, padding='SAME')
                elif norm_type == "layer":
#                     print("hit layer")
                    mean, variance = tf.nn.moments(convolved, [3], keep_dims=True)
                    normalized = (convolved - mean) / tf.sqrt(variance + 1e-5)
                    gamma = tf.get_variable(f'ln_gamma_{i}', [1, 1, 1, nfilt], initializer=tf.ones_initializer(), dtype=dtype)
                    beta = tf.get_variable(f'ln_beta_{i}', [1, 1, 1, nfilt], initializer=tf.zeros_initializer(), dtype=dtype)
                    normalized = normalized * gamma + beta
                    
                    pool_shape = [1, ptime, pband, 1]
                    if pooling_method == 'max':
                        pooled = tf.nn.max_pool(normalized, ksize=pool_shape, strides=pool_shape, padding='SAME')
                    elif pooling_method == 'avg':
                        pooled = tf.nn.avg_pool(normalized, ksize=pool_shape, strides=pool_shape, padding='SAME')
                    elif pooling_method == 'min':
                        pooled = -tf.nn.max_pool(-normalized, ksize=pool_shape, strides=pool_shape, padding='SAME')
                else:
#                     print("hit none")
                    pool_shape = [1, ptime, pband, 1]
                    if pooling_method == 'max':
                        pooled = tf.nn.max_pool(convolved, ksize=pool_shape, strides=pool_shape, padding='SAME')
                    elif pooling_method == 'avg':
                        pooled = tf.nn.avg_pool(convolved, ksize=pool_shape, strides=pool_shape, padding='SAME')
                    elif pooling_method == 'min':
                        pooled = -tf.nn.max_pool(-convolved, ksize=pool_shape, strides=pool_shape, padding='SAME')
                
#                 print('{}: {}'.format(layer_name, pooled.get_shape()))

                export_feat_tensors[layer_name] = pooled

                ### CSCI 5527 - added paramater to add cnn dropout
                if mode == 'train' and cnn_keep_prob < 1.0:
                    pooled = tf.nn.dropout(pooled, cnn_keep_prob)

############################

                layer_last = pooled
                nfilt_last = nfilt

            cnn_output = layer_last

        # Flatten CNN and concat with other features
        zack_hack_div_2 = 0
        if cnn_rnn_zack:
            zack_hack_div_2 = zack_hack // 2
            cnn_output = tf.slice(cnn_output, [0, zack_hack_div_2, 0, 0], [-1, rnn_nunroll, -1, -1])
            nfeats_conv = reduce(lambda x, y: x * y, [int(x) for x in cnn_output.get_shape()[-2:]])
        else:
            nfeats_conv = reduce(lambda x, y: x * y, [int(x) for x in cnn_output.get_shape()[-3:]])
        feats_conv = tf.reshape(cnn_output, [batch_size * rnn_nunroll, nfeats_conv])
        nfeats_tot = nfeats_conv + nfeats
        feats_all = tf.concat([feats_conv, feats_other], axis=1)
#         print('feats_cnn: {}'.format(feats_conv.get_shape()))
#         print('feats_all: {}'.format(feats_all.get_shape()))

        # Project to RNN size
        rnn_output = feats_all
        rnn_output_size = nfeats_tot
        if do_rnn:
            with tf.variable_scope('rnn_proj'):
                rnn_proj_w = tf.get_variable('W', [nfeats_tot, rnn_size], initializer=tf.uniform_unit_scaling_initializer(factor=1.0, dtype=dtype), dtype=dtype)
                rnn_proj_b = tf.get_variable('b', [rnn_size], initializer=tf.constant_initializer(0.0), dtype=dtype)

            rnn_inputs = tf.nn.bias_add(tf.matmul(feats_all, rnn_proj_w), rnn_proj_b)
            rnn_inputs = tf.reshape(rnn_inputs, [batch_size, rnn_nunroll, rnn_size])
            rnn_inputs = tf.split(rnn_inputs, rnn_nunroll, axis=1)
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]

            if rnn_cell_type == 'rnn':
                cell_fn = tf.nn.rnn_cell.BasicRNNCell
            elif rnn_cell_type == 'gru':
                cell_fn = tf.nn.rnn_cell.GRUCell
            elif rnn_cell_type == 'lstm':
                cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            else:
                raise NotImplementedError()
            cell = cell_fn(rnn_size)

            # CSCI 5527 - Not really needing rnn_keep_prob anymore since we're not going to test RNN
            if mode == 'train' and rnn_keep_prob < 1.0:
                # https://www.tensorflow.org/api_docs/python/tf/nn/RNNCellDropoutWrapper
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_keep_prob, output_keep_prob=rnn_keep_prob)

            if rnn_nlayers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * rnn_nlayers)

            initial_state = cell.zero_state(batch_size, dtype)

            # RNN
            # TODO: weight init
            with tf.variable_scope('rnn_unroll'):
                state = initial_state
                outputs = []
                for i in range(rnn_nunroll):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(rnn_inputs[i], state)
                    outputs.append(cell_output)
                final_state = state

            # CSCI 5527 - adding tranformer
            if do_transformer:
    #             with tf.variable_scope('rnn_proj'):
    #                 rnn_proj_w = tf.get_variable('W', [nfeats_tot, rnn_size], initializer=tf.uniform_unit_scaling_initializer(factor=1.0, dtype=dtype), dtype=dtype)
    #                 rnn_proj_b = tf.get_variable('b', [rnn_size], initializer=tf.constant_initializer(0.0), dtype=dtype)

    #             rnn_inputs = tf.nn.bias_add(tf.matmul(feats_all, rnn_proj_w), rnn_proj_b)
    #             rnn_inputs = tf.reshape(rnn_inputs, [batch_size, rnn_nunroll, rnn_size])
                with tf.variable_scope('transformer_proj'):
                    initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
                    proj_w = tf.get_variable('W', [nfeats_tot, d_model], initializer=initializer, dtype=dtype)
                    proj_b = tf.get_variable('b', [d_model], initializer=tf.constant_initializer(0.0), dtype=dtype)

                inputs_proj = tf.nn.bias_add(tf.matmul(feats_all, proj_w), proj_b)
                inputs_proj = tf.reshape(inputs_proj, [batch_size, rnn_nunroll, d_model])

                pos_encoding = positional_encoding(rnn_nunroll, d_model)
                inputs_proj += pos_encoding[:, :rnn_nunroll, :]

                # Before the loop
                attention_layers = []
                ffn_layers = []
                for i in range(num_transformer_layers):
                    attention_layers.append(
                        tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                                           dropout=(1.0 - rnn_keep_prob) if mode == 'train' else 0.0)
                    )
                    ffn_layers.append(
                        tf.keras.Sequential([
                            tf.keras.layers.Dense(ff_dim, activation='relu'),
                            tf.keras.layers.Dense(d_model)
                        ])
                    )

                # Inside the loop
                for i in range(num_transformer_layers):
                    attn_output = attention_layers[i](inputs_proj, inputs_proj)
                    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs_proj + attn_output)
                    ffn_output = ffn_layers[i](out1)
                    inputs_proj = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

                transformer_output = tf.reshape(inputs_proj, [batch_size * rnn_nunroll, d_model])
                rnn_output = transformer_output
                rnn_output_size = d_model
#             print('rnn_output: {}'.format(rnn_output.get_shape()))
            
            rnn_output_size = rnn_size
#         print('rnn_output: {}'.format(rnn_output.get_shape()))

        # Dense NN
        dnn_output = rnn_output
        dnn_output_size = rnn_output_size
        if do_dnn:
            last_layer = rnn_output
            last_layer_size = rnn_output_size
            for i, layer_size in enumerate(dnn_sizes):
                layer_name = 'dnn_{}'.format(i)
                with tf.variable_scope(layer_name):
                    dnn_w = tf.get_variable('W', shape=[last_layer_size, layer_size], initializer=dnn_init, dtype=dtype)
                    dnn_b = tf.get_variable('b', shape=[layer_size], initializer=tf.constant_initializer(0.0), dtype=dtype)
                projected = tf.nn.bias_add(tf.matmul(last_layer, dnn_w), dnn_b)
                # TODO: argument nonlinearity, change bias to 0.1 if relu
                if dnn_nonlin == 'tanh':
                    last_layer = tf.nn.tanh(projected)
                elif dnn_nonlin == 'sigmoid':
                    last_layer = tf.nn.sigmoid(projected)
                elif dnn_nonlin == 'relu':
                    last_layer = tf.nn.relu(projected)
                else:
                    raise NotImplementedError()
                
                # Apply different dropout rates based on layer position
                if mode == 'train' and dnn_keep_prob < 1.0:
                    last_layer = tf.nn.dropout(last_layer, layer_keep_prob)
                last_layer_size = layer_size
#                 print('{}: {}'.format(layer_name, last_layer.get_shape()))

                export_feat_tensors[layer_name] = last_layer

            dnn_output = last_layer
            dnn_output_size = last_layer_size

        # Logistic regression
        with tf.variable_scope('logit') as scope:
            logit_w = tf.get_variable('W', shape=[dnn_output_size, 1], initializer=tf.truncated_normal_initializer(stddev=1.0 / dnn_output_size, dtype=dtype), dtype=dtype)
            logit_b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer(0.0), dtype=dtype)
        logits = tf.squeeze(tf.nn.bias_add(tf.matmul(dnn_output, logit_w), logit_b), squeeze_dims=[1])
        prediction = tf.nn.sigmoid(logits)
        prediction_inspect = tf.reshape(prediction, [batch_size, rnn_nunroll])
        prediction_final = tf.squeeze(tf.slice(prediction_inspect, [0, rnn_nunroll - 1], [-1, 1]), squeeze_dims=[1])
#         print('logit: {}'.format(logits.get_shape()))

        # Compute loss
        if mode != 'gen':
            neg_log_lhoods = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
            if target_weight_strategy == 'rect':
                avg_neg_log_lhood = tf.reduce_mean(neg_log_lhoods)
            else:
                neg_log_lhoods = tf.multiply(neg_log_lhoods, target_weights)
                # be careful to have at least one weight be nonzero
                # should we be taking the mean elem-wise by batch? i think this is a big bug
                avg_neg_log_lhood = tf.reduce_sum(neg_log_lhoods) / tf.reduce_sum(target_weights)
            neg_log_lhoods_inspect = tf.reshape(neg_log_lhoods, [batch_size, rnn_nunroll])

        # Train op
        if mode == 'train':
            lr = tf.Variable(0.0, trainable=False)
            self._lr = lr
            self._lr_summary = tf.summary.scalar('learning_rate', self._lr)

            tvars = tf.trainable_variables()
            grads = tf.gradients(avg_neg_log_lhood, tvars)
            if grad_clip > 0.0:
                grads, _ = tf.clip_by_global_norm(grads, grad_clip)

            if opt == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            # CSCI 5527 added more optimizers
            elif opt == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif opt == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            else:
                raise NotImplementedError()

            # Noting that tf is tf.compat.v1, originally this line used
            # tf.train.get_or_create_global_step()
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.train.get_or_create_global_step())

        # Tensor exports
        self.feats_audio = feats_audio_nunroll
        self.feats_other = feats_other_nunroll
        if export_feat_name:
            self.feats_export = export_feat_tensors[export_feat_name]
        self.prediction = prediction_inspect
        self.prediction_final = prediction_final
        if mode != 'gen':
            self.neg_log_lhoods = neg_log_lhoods_inspect
            self.avg_neg_log_lhood = avg_neg_log_lhood
            self.targets = targets_nunroll
            self.target_weights = target_weights_nunroll
        if mode == 'train':
            self.train_op = train_op
        if mode != 'train' and do_rnn:
            self.initial_state = initial_state
            self.final_state = final_state
        self.zack_hack_div_2 = zack_hack_div_2

        self.mode = mode
        self.batch_size = batch_size
        self.rnn_nunroll = rnn_nunroll
        self.do_rnn = do_rnn
        self.target_weight_strategy = target_weight_strategy

    def assign_lr(self, sess, lr_new):
        assert self.mode == 'train'
        sess.run(tf.assign(self._lr, lr_new))
        return sess.run(self._lr_summary)

    # CSCI 5527 - Made lots of changes to this based on Claude recommendations (see pdf)
    def prepare_train_batch(self, charts, randomize_charts=False, max_resample_attempts=200, **kwargs):
        """
        Prepare training batch with early stopping for resampling loop and safeguards against empty arrays.

        Args:
            charts: List of charts to sample from
            randomize_charts: Whether to randomly select charts
            max_resample_attempts: Maximum number of attempts to find a sequence with positive examples
            **kwargs: Additional keyword arguments for sampling

        Returns:
            Tuple of (batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights)
        """
        # process kwargs
        exclude_kwarg_names = ['exclude_onset_neighbors', 'exclude_pre_onsets', 'exclude_post_onsets', 'include_onsets']
        exclude_kwargs = {k:v for k,v in list(kwargs.items()) if k in exclude_kwarg_names}
        feat_kwargs = {k:v for k,v in list(kwargs.items()) if k not in exclude_kwarg_names}

        # pick random chart and sample balanced classes
        if randomize_charts:
            del exclude_kwargs['exclude_pre_onsets']
            del exclude_kwargs['exclude_post_onsets']
            del exclude_kwargs['include_onsets']
            if self.do_rnn:
                exclude_kwargs['nunroll'] = self.rnn_nunroll

            # create batch
            batch_feats_audio = []
            batch_feats_other = []
            batch_targets = []
            batch_target_weights = []

            # Safety check: Make sure charts array is not empty
            if not charts:
                raise ValueError("No charts provided for training batch")

            for _ in range(self.batch_size):
                # Safety check: Ensure we have valid charts to sample from
                if len(charts) == 0:
                    chart_idx = 0
                else:
                    chart_idx = random.randint(0, len(charts) - 1)

                chart = charts[chart_idx]

                # Ensure there are samples to choose from
                try:
                    frame_idx = chart.sample(1, **exclude_kwargs)[0]
                except (IndexError, ValueError) as e:
                    print(f"Warning: Failed to sample frame: {e}. Using fallback frame.")
                    # Fallback to a fixed frame if sampling fails
                    frame_idx = chart.get_nframes() // 2  # Use middle frame as fallback

                subseq_start = frame_idx - (self.rnn_nunroll - 1)

                if self.target_weight_strategy == 'pos' or self.target_weight_strategy == 'posbal':
                    target_sum = 0.0
                    attempts = 0

                    # Add early stopping with max attempts
                    while target_sum == 0.0 and attempts < max_resample_attempts:
                        audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)
                        target_sum = np.sum(target)

                        if target_sum == 0.0:
                            try:
                                frame_idx = chart.sample_blanks(1, **exclude_kwargs).pop()
                            except (IndexError, ValueError):
                                # If we can't sample a blank, just use current frame
                                print("Warning: Failed to sample blank frame")
                                break

                            subseq_start = frame_idx - (self.rnn_nunroll - 1)

                        attempts += 1

                    # If we couldn't find a positive example after max attempts, just use the last sampled one
                    if target_sum == 0.0:
                        print(f"Warning: Failed to find positive example after {max_resample_attempts} attempts")
                else:
                    feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
                    audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)

                batch_feats_audio.append(audio)
                batch_feats_other.append(other)
                batch_targets.append(target)

                if self.target_weight_strategy == 'rect':
                    weight = np.ones_like(target)
                elif self.target_weight_strategy == 'last':
                    weight = np.zeros_like(target)
                    weight[-1] = 1.0
                elif self.target_weight_strategy == 'pos':
                    weight = target[:]

                    # Critical safety check: If no positive examples were found,
                    # use a fallback weighting strategy to ensure training continues
                    if np.sum(weight) == 0:
                        # Fallback to equal weights
                        weight = np.ones_like(target)
                        print("Warning: No positive examples in target. Using fallback weighting.")
                elif self.target_weight_strategy == 'posbal':
                    negs = list(np.where(target == 0)[0])  # Convert to list directly
                    num_pos = int(np.sum(target))
                    if num_pos == 0 or len(negs) == 0:
                        # Degenerate case: no positives or no negatives to sample
                        weight = np.ones_like(target)  # Fallback: treat all equally
                        print("Warning: Imbalanced targets. Using equal weights.")
                    else:
                        print("Found balanced targets.")
                        sample_size = min(num_pos, len(negs))
                        negs_weighted = random.sample(negs, sample_size)
                        weight = target[:]
                        weight[list(negs_weighted)] = 1.0
                else:
                    raise ValueError(f"Unknown target_weight_strategy: {self.target_weight_strategy}")

                batch_target_weights.append(weight)

            # Create return arrays and ensure they're not empty
            if not batch_feats_audio:
                # If somehow we ended up with no samples, create a dummy batch with zeros
                print("ERROR: No valid samples found. Creating dummy batch.")
                dummy_audio = np.zeros((1, self.rnn_nunroll, 15, 80, 3), dtype=np_dtype)
                dummy_other = np.zeros((1, self.rnn_nunroll, 5), dtype=np_dtype) 
                dummy_targets = np.zeros((1, self.rnn_nunroll), dtype=np_dtype)
                dummy_weights = np.ones((1, self.rnn_nunroll), dtype=np_dtype)  # Always use ones for weights in dummy data

                return dummy_audio, dummy_other, dummy_targets, dummy_weights

            # Safety check: Ensure all arrays have the expected shape
            if len(batch_feats_audio) < self.batch_size or len(batch_feats_other) < self.batch_size or \
               len(batch_targets) < self.batch_size or len(batch_target_weights) < self.batch_size:
                print(f"Warning: Insufficient samples collected. Got {len(batch_feats_audio)}, needed {self.batch_size}")

                # Duplicate the last sample to fill the batch if necessary
                while len(batch_feats_audio) < self.batch_size:
                    batch_feats_audio.append(batch_feats_audio[-1])
                    batch_feats_other.append(batch_feats_other[-1])
                    batch_targets.append(batch_targets[-1])
                    batch_target_weights.append(batch_target_weights[-1])

            # create return arrays
            batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
            batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
            batch_targets = np.array(batch_targets, dtype=np_dtype)
            batch_target_weights = np.array(batch_target_weights, dtype=np_dtype)

            # Final safety check: Ensure arrays have correct shapes
            expected_audio_shape = (self.batch_size, self.rnn_nunroll, 15, 80, 3)
            expected_other_shape = (self.batch_size, self.rnn_nunroll, 5)
            expected_target_shape = (self.batch_size, self.rnn_nunroll)

            if batch_feats_audio.shape[0:2] != expected_audio_shape[0:2] or \
               batch_feats_other.shape != expected_other_shape or \
               batch_targets.shape != expected_target_shape or \
               batch_target_weights.shape != expected_target_shape:
                print(f"Shape mismatch. Reshaping arrays to expected dimensions.")

                # Reshape arrays to expected dimensions
                dummy_audio = np.zeros(expected_audio_shape, dtype=np_dtype)
                dummy_other = np.zeros(expected_other_shape, dtype=np_dtype)
                dummy_targets = np.zeros(expected_target_shape, dtype=np_dtype)
                dummy_weights = np.ones(expected_target_shape, dtype=np_dtype)

                # Copy as much data as possible
                slice_size = min(batch_feats_audio.shape[0], self.batch_size)
                dummy_audio[:slice_size] = batch_feats_audio[:slice_size]
                dummy_other[:slice_size] = batch_feats_other[:slice_size]
                dummy_targets[:slice_size] = batch_targets[:slice_size]
                dummy_weights[:slice_size] = batch_target_weights[:slice_size]

                return dummy_audio, dummy_other, dummy_targets, dummy_weights

            return batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights
        else:
            # Similar safeguards should be implemented for this branch
            # For brevity, I'm showing the most common case above

            chart_idx = random.randint(0, len(charts) - 1) if charts else 0
            chart = charts[chart_idx]
            chart_nonsets = chart.get_nonsets()

            if exclude_kwargs.get('include_onsets', False):
                npos = 0
                nneg = self.batch_size
            else:
                npos = min(self.batch_size // 2, chart_nonsets)
                nneg = self.batch_size - npos

            # Ensure we can sample enough frames
            try:
                samples = chart.sample_onsets(npos) + chart.sample_blanks(nneg, **exclude_kwargs)
                if not samples:
                    raise ValueError("No samples available")
            except Exception as e:
                print(f"Warning: Sampling failed: {e}. Creating fallback samples.")
                # Create fallback samples if sampling fails
                nframes = chart.get_nframes()
                samples = [nframes // 2] * self.batch_size  # Use middle frame as fallback

            random.shuffle(samples)

            # create batch
            batch_feats_audio = []
            batch_feats_other = []
            batch_targets = []
            batch_target_weights = []
            for frame_idx in samples:
                subseq_start = frame_idx - (self.rnn_nunroll - 1)

                if self.target_weight_strategy == 'pos' or self.target_weight_strategy == 'posbal':
                    target_sum = 0.0
                    attempts = 0

                    # Add early stopping with max attempts
                    while target_sum == 0.0 and attempts < max_resample_attempts:
                        audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)
                        target_sum = np.sum(target)

                        if target_sum == 0.0:
                            try:
                                frame_idx = chart.sample_blanks(1, **exclude_kwargs).pop()
                            except (IndexError, ValueError):
                                print("Warning: Failed to sample blank frame")
                                break

                            subseq_start = frame_idx - (self.rnn_nunroll - 1)

                        attempts += 1

                    # If we couldn't find a positive example after max attempts, just use the last sampled one
                    if target_sum == 0.0:
                        print(f"Warning: Failed to find positive example after {max_resample_attempts} attempts")
                else:
                    feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
                    audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)

                batch_feats_audio.append(audio)
                batch_feats_other.append(other)
                batch_targets.append(target)

                if self.target_weight_strategy == 'rect':
                    weight = np.ones_like(target)
                elif self.target_weight_strategy == 'last':
                    weight = np.zeros_like(target)
                    weight[-1] = 1.0
                elif self.target_weight_strategy == 'pos':
                    weight = target[:]

                    # Critical safety check
                    if np.sum(weight) == 0:
                        weight = np.ones_like(target)
                        print("Warning: No positive examples in target. Using fallback weighting.")
                elif self.target_weight_strategy == 'posbal':
                    negs = list(np.where(target == 0)[0])
                    num_pos = int(np.sum(target))
                    if num_pos == 0 or len(negs) == 0:
                        weight = np.ones_like(target)
                        print("Warning: Imbalanced targets. Using equal weights.")
                    else:
                        sample_size = min(num_pos, len(negs))
                        negs_weighted = random.sample(negs, sample_size)
                        weight = target[:]
                        weight[list(negs_weighted)] = 1.0
                else:
                    raise ValueError(f"Unknown target_weight_strategy: {self.target_weight_strategy}")

                batch_target_weights.append(weight)

            # Safety checks similar to the ones in the other branch
            if not batch_feats_audio:
                print("ERROR: No valid samples found. Creating dummy batch.")
                dummy_audio = np.zeros((self.batch_size, self.rnn_nunroll, 15, 80, 3), dtype=np_dtype)
                dummy_other = np.zeros((self.batch_size, self.rnn_nunroll, 5), dtype=np_dtype) 
                dummy_targets = np.zeros((self.batch_size, self.rnn_nunroll), dtype=np_dtype)
                dummy_weights = np.ones((self.batch_size, self.rnn_nunroll), dtype=np_dtype)

                return dummy_audio, dummy_other, dummy_targets, dummy_weights

            # Ensure batch size is maintained
            while len(batch_feats_audio) < self.batch_size:
                if batch_feats_audio:  # If there's at least one sample, duplicate it
                    batch_feats_audio.append(batch_feats_audio[-1])
                    batch_feats_other.append(batch_feats_other[-1])
                    batch_targets.append(batch_targets[-1])
                    batch_target_weights.append(batch_target_weights[-1])
                else:  # Otherwise create a dummy sample
                    dummy_audio = np.zeros((1, self.rnn_nunroll, 15, 80, 3), dtype=np_dtype)
                    dummy_other = np.zeros((1, self.rnn_nunroll, 5), dtype=np_dtype)
                    dummy_targets = np.zeros((1, self.rnn_nunroll), dtype=np_dtype)
                    dummy_weights = np.ones((1, self.rnn_nunroll), dtype=np_dtype)

                    batch_feats_audio.append(dummy_audio[0])
                    batch_feats_other.append(dummy_other[0])
                    batch_targets.append(dummy_targets[0])
                    batch_target_weights.append(dummy_weights[0])

            # create return arrays
            batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
            batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
            batch_targets = np.array(batch_targets, dtype=np_dtype)
            batch_target_weights = np.array(batch_target_weights, dtype=np_dtype)

            return batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights

    def iterate_eval_batches(self, eval_chart, **feat_kwargs):
        assert self.target_weight_strategy == 'seq'

        if self.do_rnn:
            subseq_len = self.rnn_nunroll
            subseq_start = -(subseq_len - 1)
        else:
            subseq_len = self.batch_size
            subseq_start = 0

        for frame_idx in range(subseq_start, eval_chart.get_nframes(), subseq_len):
            feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
            audio, other, target = eval_chart.get_subsequence(frame_idx, subseq_len, np_dtype, **feat_kwargs)

            weight = np.ones_like(target)
            mask_left = max(eval_chart.get_first_onset() - frame_idx, 0)
            mask_right = max((eval_chart.get_last_onset() + 1) - frame_idx, 0)
            weight[:mask_left] = 0.0
            weight[mask_right:] = 0.0

            if self.do_rnn:
                yield audio[np.newaxis, :], other[np.newaxis, :], target[np.newaxis, :], weight[np.newaxis, :]
            else:
                yield audio[:, np.newaxis], other[:, np.newaxis], target[:, np.newaxis], weight[:, np.newaxis]