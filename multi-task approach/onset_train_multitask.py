from collections import defaultdict
import pickle as picklmaine
import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score

from onset_net import OnsetNet
from util import *

import argparse

def parse_args(get_defaults=False):
    parser = argparse.ArgumentParser(description='OnsetNet Training Script')

    # Data
    parser.add_argument('--train_txt_fp', type=str, default='')
    parser.add_argument('--valid_txt_fp', type=str, default='')
    parser.add_argument('--z_score', action='store_true')
    parser.add_argument('--test_txt_fp', type=str, default='')
    parser.add_argument('--model_ckpt_fp', type=str, default='')
    parser.add_argument('--export_feat_name', type=str, default='')

    # Features
    parser.add_argument('--audio_context_radius', type=int, default=7)
    parser.add_argument('--audio_nbands', type=int, default=80)
    parser.add_argument('--audio_nchannels', type=int, default=3)
    parser.add_argument('--audio_select_channels', type=str, default='')
    parser.add_argument('--feat_diff_feet_to_id_fp', type=str, default='')
    parser.add_argument('--feat_diff_coarse_to_id_fp', type=str, default='')
    parser.add_argument('--feat_diff_dipstick', action='store_true')
    parser.add_argument('--feat_freetext_to_id_fp', type=str, default='')
    parser.add_argument('--feat_beat_phase', action='store_true')
    parser.add_argument('--feat_beat_phase_cos', action='store_true')

    # Network params
    parser.add_argument('--cnn_filter_shapes', type=str, default='')
    parser.add_argument('--cnn_pool', type=str, default='')
    parser.add_argument('--cnn_rnn_zack', action='store_true')
    parser.add_argument('--zack_hack', type=int, default=0)
    parser.add_argument('--rnn_cell_type', type=str, default='lstm')
    parser.add_argument('--rnn_size', type=int, default=0)
    parser.add_argument('--rnn_nlayers', type=int, default=0)
    parser.add_argument('--rnn_nunroll', type=int, default=1)
    parser.add_argument('--rnn_keep_prob', type=float, default=1.0)
    parser.add_argument('--dnn_sizes', type=str, default='')
    parser.add_argument('--dnn_keep_prob', type=float, default=1.0)
    parser.add_argument('--dnn_nonlin', type=str, default='sigmoid')
    parser.add_argument('--pooling_method', type=str, default='max')

    # Multi-task learning params
    parser.add_argument('--enable_multi_task', action='store_true', help='Enable multi-task learning')
    parser.add_argument('--task_weights', type=str, default='0.7,0.3', help='Comma-separated weights for each task')

    # Training params
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_strategy', type=str, default='rect')
    parser.add_argument('--randomize_charts', action='store_true')
    parser.add_argument('--balanced_class', action='store_true')
    parser.add_argument('--exclude_onset_neighbors', type=int, default=0)
    parser.add_argument('--exclude_pre_onsets', action='store_true')
    parser.add_argument('--exclude_post_onsets', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--norm_type', type=str, default='none')
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--lr_decay_rate', type=float, default=1.0)
    parser.add_argument('--lr_decay_delay', type=int, default=0)
    parser.add_argument('--nbatches_per_ckpt', type=int, default=100)
    parser.add_argument('--nbatches_per_eval', type=int, default=10000)
    parser.add_argument('--nepochs', type=int, default=0)
    parser.add_argument('--experiment_dir', type=str, default='')

    # Eval params
    parser.add_argument('--eval_window_type', type=str, default='')
    parser.add_argument('--eval_window_width', type=int, default=0)
    parser.add_argument('--eval_align_tolerance', type=int, default=2)
    parser.add_argument('--eval_charts_export', type=str, default='')
    parser.add_argument('--eval_diff_coarse', type=str, default='')
    
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--ff_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    
    if get_defaults:
        return parser.parse_args([])

    return parser.parse_args()

dtype = tf.float32

def train_onset_model(args, OnsetModel=OnsetNet):
    assert args.experiment_dir
    do_train = args.nepochs != 0 and bool(args.train_txt_fp)
    do_valid = bool(args.valid_txt_fp)
    do_train_eval = do_train and do_valid
    do_eval = bool(args.test_txt_fp)
    do_cnn_export = bool(args.export_feat_name)
    
    # Check for multi-task learning
    enable_multi_task = getattr(args, 'enable_multi_task', False)
    task_weights = None
    if enable_multi_task and hasattr(args, 'task_weights'):
        # Parse task weights from string like "0.7,0.3"
        if isinstance(args.task_weights, str) and args.task_weights:
            task_weights = [float(w) for w in args.task_weights.split(',')]
        elif isinstance(args.task_weights, (list, tuple)):
            task_weights = list(args.task_weights)
    
    # Load data
    print('Loading data')
    train_data, valid_data, test_data = open_dataset_fps(args.train_txt_fp, args.valid_txt_fp, args.test_txt_fp)

    # Select channels
    if args.audio_select_channels:
        channels = stride_csv_arg_list(args.audio_select_channels, 1, int)
        print('Selecting channels {} from data'.format(channels))
        for data in [train_data, valid_data, test_data]:
            select_channels(data, channels)

    # Calculate validation metrics
    if args.z_score:
        z_score_fp = os.path.join(args.experiment_dir, 'valid_mean_std.pkl')
        if do_valid and not os.path.exists(z_score_fp):
            print('Calculating validation metrics')
            mean_per_band, std_per_band = calc_mean_std_per_band(valid_data)
            with open(z_score_fp, 'wb') as f:
                pickle.dump((mean_per_band, std_per_band), f)
        else:
            print('Loading validation metrics')
            with open(z_score_fp, 'rb') as f:
                mean_per_band, std_per_band = pickle.load(f)

        # Sanitize data
        for data in [train_data, valid_data, test_data]:
            apply_z_norm(data, mean_per_band, std_per_band)

    # Flatten the data into chart references for easier iteration
    print('Flattening datasets into charts')
    charts_train = flatten_dataset_to_charts(train_data)
    charts_valid = flatten_dataset_to_charts(valid_data)
    charts_test = flatten_dataset_to_charts(test_data)
    print('Train set: {} charts, valid set: {} charts, test set: {} charts'.format(len(charts_train), len(charts_valid), len(charts_test)))

    # Load ID maps
    diff_feet_to_id = None
    if args.feat_diff_feet_to_id_fp:
        diff_feet_to_id = load_id_dict(args.feat_diff_feet_to_id_fp)
    diff_coarse_to_id = None
    if args.feat_diff_coarse_to_id_fp:
        diff_coarse_to_id = load_id_dict(args.feat_diff_coarse_to_id_fp)
    freetext_to_id = None
    if args.feat_freetext_to_id_fp:
        freetext_to_id = load_id_dict(args.feat_freetext_to_id_fp)

    # Create feature config
    feats_config = {
        'time_context_radius': args.audio_context_radius,
        'diff_feet_to_id': diff_feet_to_id,
        'diff_coarse_to_id': diff_coarse_to_id,
        'diff_dipstick': args.feat_diff_dipstick,
        'freetext_to_id': freetext_to_id,
        'beat_phase': args.feat_beat_phase,
        'beat_phase_cos': args.feat_beat_phase_cos
    }
    nfeats = 0
    nfeats += 0 if diff_feet_to_id is None else max(diff_feet_to_id.values()) + 1
    nfeats += 0 if diff_coarse_to_id is None else max(diff_coarse_to_id.values()) + 1
    nfeats += 0 if freetext_to_id is None else max(freetext_to_id.values()) + 1
    nfeats += 1 if args.feat_beat_phase else 0
    nfeats += 1 if args.feat_beat_phase_cos else 0
    print('Feature configuration (nfeats={}): {}'.format(nfeats, feats_config))

    # Create training data exclusions config
    tn_exclusions = {
        'randomize_charts': args.randomize_charts,
        'exclude_onset_neighbors': args.exclude_onset_neighbors,
        'exclude_pre_onsets': args.exclude_pre_onsets,
        'exclude_post_onsets': args.exclude_post_onsets,
        'include_onsets': not args.balanced_class
    }
    train_batch_config = feats_config.copy()
    train_batch_config.update(tn_exclusions)
    print('Exclusions: {}'.format(tn_exclusions))

    # Create model config
    model_config = {
        'audio_context_radius': args.audio_context_radius,
        'audio_nbands': args.audio_nbands,
        'audio_nchannels': args.audio_nchannels,
        'nfeats': nfeats,
        'cnn_filter_shapes': stride_csv_arg_list(args.cnn_filter_shapes, 3, int),
        'cnn_init': tf.uniform_unit_scaling_initializer(factor=1.43, dtype=dtype),
        'cnn_pool': stride_csv_arg_list(args.cnn_pool, 2, int),
        'cnn_rnn_zack': args.cnn_rnn_zack,
        'zack_hack': args.zack_hack,
        'rnn_cell_type': args.rnn_cell_type,
        'rnn_size': args.rnn_size,
        'rnn_nlayers': args.rnn_nlayers,
        'rnn_init': tf.random_uniform_initializer(-5e-2, 5e-2, dtype=dtype),
        'rnn_nunroll': args.rnn_nunroll,
        'rnn_keep_prob': args.rnn_keep_prob,
        'dnn_sizes': stride_csv_arg_list(args.dnn_sizes, 1, int),
        'dnn_init': tf.uniform_unit_scaling_initializer(factor=1.15, dtype=dtype),
        'dnn_keep_prob': args.dnn_keep_prob,
        'dnn_nonlin': args.dnn_nonlin,
        'pooling_method': args.pooling_method,
        'grad_clip': args.grad_clip,
        'opt': args.opt,
        'norm_type': args.norm_type,
        'd_model': args.d_model,
        'num_transformer_layers': args.num_transformer_layers,
        'num_heads': args.num_heads,
        'ff_dim': args.ff_dim
    }
    print('Model configuration: {}'.format(model_config))

    with tf.Graph().as_default(), tf.Session() as sess:
        if do_train:
            print('Creating train model')
            with tf.variable_scope('model_sp', reuse=None):
                model_train = OnsetModel(
                    mode='train', 
                    target_weight_strategy=args.weight_strategy, 
                    batch_size=args.batch_size, 
                    enable_multi_task=enable_multi_task,
                    task_weights=task_weights,
                    **model_config
                )

        if do_train_eval or do_eval or do_cnn_export:
            with tf.variable_scope('model_sp', reuse=do_train):
                eval_batch_size = args.batch_size
                if args.rnn_nunroll > 1:
                    eval_batch_size = 1
                model_eval = OnsetModel(
                    mode='eval', 
                    target_weight_strategy='seq', 
                    batch_size=eval_batch_size, 
                    export_feat_name=args.export_feat_name,
                     enable_multi_task=enable_multi_task,
                    task_weights=task_weights,
                    **model_config
                )
                model_early_stop_xentropy_avg = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                model_early_stop_auprc = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                model_early_stop_fscore = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # Restore or init model
        model_saver = tf.train.Saver(tf.global_variables())
        if args.model_ckpt_fp:
            print('Restoring model weights from {}'.format(args.model_ckpt_fp))
            model_saver.restore(sess, args.model_ckpt_fp)
        else:
            print('Initializing model weights from scratch')
            sess.run(tf.global_variables_initializer())

        # Create summaries
        if do_train:
            summary_writer = tf.summary.FileWriter(args.experiment_dir, sess.graph)

            epoch_mean_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_mean_xentropy')
            epoch_mean_time = tf.placeholder(tf.float32, shape=[], name='epoch_mean_time')
            epoch_var_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_var_xentropy')
            epoch_var_time = tf.placeholder(tf.float32, shape=[], name='epoch_var_time')
            epoch_summaries = tf.summary.merge([
                tf.summary.scalar('epoch_mean_xentropy', epoch_mean_xentropy),
                tf.summary.scalar('epoch_mean_time', epoch_mean_time),
                tf.summary.scalar('epoch_var_xentropy', epoch_var_xentropy),
                tf.summary.scalar('epoch_var_time', epoch_var_time)
            ])

            eval_metric_names = ['xentropy_avg', 'pos_xentropy_avg', 'auroc', 'auprc', 'fscore', 'precision', 'recall', 'threshold', 'accuracy', 'perplexity', 'density_rel']
            eval_metrics = {}
            eval_summaries = []
            for eval_metric_name in eval_metric_names:
                name_mean = 'eval_mean_{}'.format(eval_metric_name)
                name_var = 'eval_var_{}'.format(eval_metric_name)
                ph_mean = tf.placeholder(tf.float32, shape=[], name=name_mean)
                ph_var = tf.placeholder(tf.float32, shape=[], name=name_var)
                summary_mean = tf.summary.scalar(name_mean, ph_mean)
                summary_var = tf.summary.scalar(name_var, ph_var)
                eval_summaries.append(tf.summary.merge([summary_mean, summary_var]))
                eval_metrics[eval_metric_name] = (ph_mean, ph_var)
            eval_time = tf.placeholder(tf.float32, shape=[], name='eval_time')
            eval_time_summary = tf.summary.scalar('eval_time', eval_time)
            eval_summaries = tf.summary.merge([eval_time_summary] + eval_summaries)

            # Calculate epoch stuff
            train_nframes = sum([chart.get_nframes_annotated() for chart in charts_train])
            examples_per_batch = args.batch_size
            examples_per_batch *= args.rnn_nunroll if args.weight_strategy == 'rect' else 1
            batches_per_epoch = train_nframes // examples_per_batch
            nbatches = args.nepochs * batches_per_epoch
            print('{} frames in data, {} batches per epoch, {} batches total'.format(train_nframes, batches_per_epoch, nbatches))

            # Init epoch
            lr_summary = model_train.assign_lr(sess, args.lr)
            summary_writer.add_summary(lr_summary, 0)
            epoch_xentropies = []
            epoch_times = []

            batch_num = 0
            eval_best_xentropy_avg = float('inf')
            eval_best_auprc = 0.0
            eval_best_fscore = 0.0
            while args.nepochs < 0 or batch_num < nbatches:
                batch_time_start = time.time()
                feats_audio, feats_other, targets, target_weights = model_train.prepare_train_batch(charts_train, **train_batch_config)
                feed_dict = {
                    model_train.feats_audio: feats_audio,
                    model_train.feats_other: feats_other,
                    model_train.targets: targets,
                    model_train.target_weights: target_weights
                }
                
                # Add additional feed dict entries for multi-task
                if model_train.enable_multi_task:
                    # Calculate density targets - count steps in a window ahead
                    density_window = 10  # Keep consistent with model definition
                    density_targets = np.zeros_like(targets.flatten())
                    
                    # For each batch item
                    flat_targets = targets.flatten()
                    for i in range(len(flat_targets) - density_window):
                        # Count steps in the next density_window frames
                        density_targets[i] = np.sum(flat_targets[i:i+density_window])
                    
                    # For the last few frames, use what's available
                    for i in range(len(flat_targets) - density_window, len(flat_targets)):
                        density_targets[i] = np.sum(flat_targets[i:])
                    
                    feed_dict[model_train.density_targets] = density_targets
                
                batch_xentropy, _ = sess.run([model_train.avg_neg_log_lhood, model_train.train_op], feed_dict=feed_dict)

                epoch_xentropies.append(batch_xentropy)
                epoch_times.append(time.time() - batch_time_start)

                batch_num += 1

                if batch_num % batches_per_epoch == 0:
                    epoch_num = batch_num // batches_per_epoch
                    print('Completed epoch {}'.format(epoch_num))

                    lr_decay = args.lr_decay_rate ** max(epoch_num - args.lr_decay_delay, 0)
                    lr_summary = model_train.assign_lr(sess, args.lr * lr_decay)
                    summary_writer.add_summary(lr_summary, batch_num)

                    epoch_xentropy = np.mean(epoch_xentropies)
                    print('Epoch mean cross-entropy (nats) {}'.format(epoch_xentropy))
                    epoch_summary = sess.run(epoch_summaries, feed_dict={epoch_mean_xentropy: epoch_xentropy, epoch_mean_time: np.mean(epoch_times), epoch_var_xentropy: np.var(epoch_xentropies), epoch_var_time: np.var(epoch_times)})
                    summary_writer.add_summary(epoch_summary, batch_num)

                    epoch_xentropies = []
                    epoch_times = []

                if batch_num % args.nbatches_per_ckpt == 0:
                    print('Saving model weights...')
                    ckpt_fp = os.path.join(args.experiment_dir, 'onset_net_train')
                    model_saver.save(sess, ckpt_fp, global_step=tf.train.get_or_create_global_step())
                    print('Done saving!')

                if do_train_eval and batch_num % args.nbatches_per_eval == 0:
                    print('Evaluating...')
                    eval_start_time = time.time()

                    metrics = defaultdict(list)
                    for chart in charts_valid:
                        y_true, y_scores, y_xentropies, y_scores_pkalgn = model_scores_for_chart(sess, chart, model_eval, args, **feats_config)
                        assert int(np.sum(y_true)) == chart.get_nonsets()

                        chart_metrics = eval_metrics_for_scores(y_true, y_scores, y_xentropies, y_scores_pkalgn)
                        for metrics_key, metric_value in list(chart_metrics.items()):
                            metrics[metrics_key].append(metric_value)

                    metrics = {k: (np.mean(v), np.var(v)) for k, v in list(metrics.items())}
                    feed_dict = {}
                    results = []
                    for metric_name, (mean, var) in list(metrics.items()):
                        feed_dict[eval_metrics[metric_name][0]] = mean
                        feed_dict[eval_metrics[metric_name][1]] = var
                    feed_dict[eval_time] = time.time() - eval_start_time

                    summary_writer.add_summary(sess.run(eval_summaries, feed_dict=feed_dict), batch_num)

                    xentropy_avg_mean = metrics['xentropy_avg'][0]
                    if xentropy_avg_mean < eval_best_xentropy_avg:
                        print('Xentropy {} better than previous {}'.format(xentropy_avg_mean, eval_best_xentropy_avg))
                        ckpt_fp = os.path.join(args.experiment_dir, 'onset_net_early_stop_xentropy_avg')
                        model_early_stop_xentropy_avg.save(sess, ckpt_fp, global_step=tf.train.get_or_create_global_step())
                        eval_best_xentropy_avg = xentropy_avg_mean

                    auprc_mean = metrics['auprc'][0]
                    if auprc_mean > eval_best_auprc:
                        print('AUPRC {} better than previous {}'.format(auprc_mean, eval_best_auprc))
                        ckpt_fp = os.path.join(args.experiment_dir, 'onset_net_early_stop_auprc')
                        model_early_stop_auprc.save(sess, ckpt_fp, global_step=tf.train.get_or_create_global_step())
                        eval_best_auprc = auprc_mean

                    fscore_mean = metrics['fscore'][0]
                    if fscore_mean > eval_best_fscore:
                        print('Fscore {} better than previous {}'.format(fscore_mean, eval_best_fscore))
                        ckpt_fp = os.path.join(args.experiment_dir, 'onset_net_early_stop_fscore')
                        model_early_stop_fscore.save(sess, ckpt_fp, global_step=tf.train.get_or_create_global_step())
                        eval_best_fscore = fscore_mean

                    print('Done evaluating')

        if do_cnn_export:
            print('Exporting CNN features...')
            export_dir = os.path.join(args.experiment_dir, 'export_{}'.format(args.export_feat_name))

            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # This is ugly... Deal with it
            song_names = []
            for data_fp in [args.train_txt_fp, args.valid_txt_fp, args.test_txt_fp]:
                with open(data_fp, 'r') as f:
                    song_charts_fps = f.read().splitlines()
                song_names += [os.path.splitext(os.path.split(x)[1])[0] for x in song_charts_fps]

            songs_complete = set()
            for charts in [charts_train, charts_valid, charts_test]:
                for chart in charts:
                    song_features_id = id(chart.song_features)
                    if song_features_id in songs_complete:
                        continue

                    song_feats_export = []
                    for feats_audio, feats_other, _, _ in model_eval.iterate_eval_batches(chart, **feats_config):
                        assert feats_other.shape[2] == 0
                        feed_dict = {
                            model_eval.feats_audio: feats_audio,
                            model_eval.feats_other: feats_other,
                        }
                        feats_export = sess.run(model_eval.feats_export, feed_dict=feed_dict)
                        song_feats_export.append(feats_export)

                    song_feats_export = np.concatenate(song_feats_export)
                    song_feats_export = song_feats_export[:chart.song_features.shape[0]]
                    assert song_feats_export.shape[0] == chart.song_features.shape[0]

                    if 'cnn' in args.export_feat_name:
                        song_feats_export = np.reshape(song_feats_export, (song_feats_export.shape[0], -1, song_feats_export.shape[3]))
                    if 'dnn' in args.export_feat_name:
                        song_feats_export = song_feats_export[:, :, np.newaxis]

                    assert song_feats_export.ndim == 3

                    out_name = song_names.pop(0)
                    print('{} ({})->{} ({})'.format(chart.get_song_metadata(), chart.song_features.shape, out_name, song_feats_export.shape))

                    with open(os.path.join(export_dir, '{}.pkl'.format(out_name)), 'wb') as f:
                        pickle.dump(song_feats_export, f)

                    songs_complete.add(song_features_id)

            assert len(song_names) == 0

        if do_eval and not do_cnn_export:
            print('Evaluating...')

            exports = stride_csv_arg_list(args.eval_charts_export, 1, int)
            diff_concat = lambda d, diff: np.concatenate(d[diff])
            diff_array = lambda d, diff: np.array(d[diff])
            all_concat = lambda d: np.concatenate([diff_concat(d, diff) for diff in diffs])
            all_concat_array = lambda d: np.concatenate([diff_array(d, diff) for diff in diffs])

            # calculate thresholds
            if len(charts_valid) > 0:
                print('Calculating perchart and micro thresholds for validation data')

                # go through charts calculating scores and thresholds
                diff_to_threshold = defaultdict(list)
                diff_to_y_true_all = defaultdict(list)
                diff_to_y_scores_pkalgn_all = defaultdict(list)
                for chart in charts_valid:
                    y_true, _, _, y_scores_pkalgn = model_scores_for_chart(sess, chart, model_eval, args, **feats_config)
                    chart_metrics = eval_metrics_for_scores(y_true, None, None, y_scores_pkalgn)

                    diff_to_threshold[chart.get_coarse_difficulty()].append(chart_metrics['threshold'])
                    diff_to_y_true_all[chart.get_coarse_difficulty()].append(y_true)
                    diff_to_y_scores_pkalgn_all[chart.get_coarse_difficulty()].append(y_scores_pkalgn)

                # lambdas for calculating micro thresholds
                diffs = list(diff_to_y_true_all.keys())

                # calculate diff perchart
                diff_to_threshold_perchart = {diff:np.mean(diff_array(diff_to_threshold, diff)) for diff in diffs}

                # calculate all perchart
                threshold_all_perchart = np.mean(all_concat_array(diff_to_threshold))

                # calculate diff micro
                diff_to_threshold_micro = {}
                for diff in diffs:
                    diff_metrics = eval_metrics_for_scores(diff_concat(diff_to_y_true_all, diff), None, None, diff_concat(diff_to_y_scores_pkalgn_all, diff))
                    diff_to_threshold_micro[diff] = diff_metrics['threshold']

                # calculate all micro
                all_metrics = eval_metrics_for_scores(all_concat(diff_to_y_true_all), None, None, all_concat(diff_to_y_scores_pkalgn_all))
                threshold_all_micro = all_metrics['threshold']

                print('Diff perchart thresholds: {}'.format(diff_to_threshold_perchart))
                print('All perchart threshold: {}'.format(threshold_all_perchart))
                print('Diff_micro thresholds: {}'.format(diff_to_threshold_micro))
                print('All micro thresholds: {}'.format(threshold_all_micro))

            # run evaluation on test data
            diff_to_y_true = defaultdict(list)
            diff_to_y_scores = defaultdict(list)
            diff_to_y_xentropies = defaultdict(list)
            diff_to_y_scores_pkalgn = defaultdict(list)
            metrics = defaultdict(list)
            for i, chart in enumerate(charts_test):
                chart_coarse = chart.get_coarse_difficulty()
                if args.eval_diff_coarse:
                    if chart_coarse != args.eval_diff_coarse:
                        continue

                y_true, y_scores, y_xentropies, y_scores_pkalgn = model_scores_for_chart(sess, chart, model_eval, args, **feats_config)
                assert int(np.sum(y_true)) == chart.get_nonsets()
                diff_to_y_true[chart_coarse].append(y_true)
                diff_to_y_scores[chart_coarse].append(y_scores)
                diff_to_y_xentropies[chart_coarse].append(y_xentropies)
                diff_to_y_scores_pkalgn[chart_coarse].append(y_scores_pkalgn)

                if i in exports:
                    chart_name = ez_name(chart.get_song_metadata()['title'])
                    chart_export_fp = os.path.join(args.experiment_dir, '{}_{}.pkl'.format(chart_name, chart.get_foot_difficulty()))
                    chart_eval_save = {
                        'song_metadata': chart.get_song_metadata(),
                        'song_feats': chart.song_features[:, :, 0],
                        'chart_feet': chart.get_foot_difficulty(),
                        'chart_onsets': chart.get_onsets(),
                        'y_true': y_true,
                        'y_scores': y_scores,
                        'y_xentropies': y_xentropies,
                        'y_scores_pkalgn': y_scores_pkalgn
                    }
                    with open(chart_export_fp, 'wb') as f:
                        print('Saving {} {}'.format(chart.get_song_metadata(), chart.get_foot_difficulty()))
                        pickle.dump(chart_eval_save, f)

                if len(charts_valid) > 0:
                    threshold_names = ['diff_perchart', 'all_perchart', 'diff_micro', 'all_micro']
                    thresholds = [diff_to_threshold_perchart[chart_coarse], threshold_all_perchart, diff_to_threshold_micro[chart_coarse], threshold_all_micro]
                else:
                    threshold_names, thresholds = [], []

                chart_metrics = eval_metrics_for_scores(y_true, y_scores, y_xentropies, y_scores_pkalgn, threshold_names, thresholds)
                for metrics_key, metric in list(chart_metrics.items()):
                    metrics[metrics_key].append(metric)

            # calculate micro metrics
            diffs = list(diff_to_y_true.keys())
            metrics_micro, prc = eval_metrics_for_scores(all_concat(diff_to_y_true), all_concat(diff_to_y_scores), all_concat(diff_to_y_xentropies), all_concat(diff_to_y_scores_pkalgn), return_prc=True)

            # dump PRC for inspection
            with open(os.path.join(args.experiment_dir, 'prc.pkl'), 'wb') as f:
                pickle.dump(prc, f)

            # calculate perchart metrics
            metrics_perchart = {k: (np.mean(v), np.std(v), np.min(v), np.max(v)) for k, v in list(metrics.items())}

            # report metrics
            copy_pasta = []

            metrics_report_micro = ['xentropy_avg', 'pos_xentropy_avg', 'auroc', 'auprc', 'fscore', 'precision', 'recall', 'threshold', 'accuracy']
            for metric_name in metrics_report_micro:
                metric_stats = metrics_micro[metric_name]
                copy_pasta += [metric_stats]
                print('micro_{}: {}'.format(metric_name, metric_stats))

            metrics_report_perchart = ['xentropy_avg', 'pos_xentropy_avg', 'auroc', 'auprc', 'fscore', 'precision', 'recall', 'threshold', 'accuracy', 'perplexity', 'density_rel']
            for threshold_name in threshold_names:
                fmetric_names = ['fscore_{}', 'precision_{}', 'recall_{}', 'threshold_{}']
                metrics_report_perchart += [name.format(threshold_name) for name in fmetric_names]
            for metric_name in metrics_report_perchart:
                metric_stats = metrics_perchart.get(metric_name, (0., 0., 0., 0.))
                copy_pasta += list(metric_stats)
                print('perchart_{}: {}'.format(metric_name, metric_stats))

            print('COPY PASTA:')
            print(','.join([str(x) for x in copy_pasta]))

def model_scores_for_chart(sess, chart, model, args, **feat_kwargs):
    if model.do_rnn:
        state = sess.run(model.initial_state)
    targets_all = []
    scores = []
    xentropies = []
    weight_sum = 0.0
    target_sum = 0.0

    chunk_len = args.rnn_nunroll if model.do_rnn else args.batch_size
    for feats_audio, feats_other, targets, target_weights in model.iterate_eval_batches(chart, **feat_kwargs):
        feed_dict = {
            model.feats_audio: feats_audio,
            model.feats_other: feats_other,
            model.targets: targets,
            model.target_weights: target_weights
        }
        if model.do_rnn:
            feed_dict[model.initial_state] = state
            state, seq_scores, seq_xentropies = sess.run([model.final_state, model.prediction, model.neg_log_lhoods], feed_dict=feed_dict)
            scores.append(seq_scores[0])
            xentropies.append(seq_xentropies[0])
        else:
            seq_scores, seq_xentropies = sess.run([model.prediction, model.neg_log_lhoods], feed_dict=feed_dict)
            scores.append(seq_scores[:, 0])
            xentropies.append(seq_xentropies[:, 0])

        targets_all.append(targets)
        weight_sum += np.sum(target_weights)
        target_sum += np.sum(targets)

    targets_all = np.concatenate(targets_all)
    assert int(weight_sum) == chart.get_nframes_annotated()
    assert int(target_sum) == chart.get_nonsets()

    # scores may be up to nunroll-1 longer than song feats but will be left-aligned
    scores = np.concatenate(scores)
    xentropies = np.concatenate(xentropies)
    assert scores.shape[0] >= chart.get_nframes()
    assert scores.shape[0] < (chart.get_nframes() + 2 * chunk_len)
    assert xentropies.shape == scores.shape
    if model.do_rnn:
        xentropies = xentropies[chunk_len - 1:]
        scores = scores[chunk_len - 1:]
    scores = scores[:chart.get_nframes()]

    # find predicted onsets (smooth and peak pick)
    if args.eval_window_type == 'hann':
        window = np.hanning(args.eval_window_width)
    elif args.eval_window_type == 'hamming':
        window = np.hamming(args.eval_window_width)
    else:
        raise NotImplementedError()
    pred_onsets = find_pred_onsets(scores, window)

    # align scores with true to create sklearn-compatible vectors
    true_onsets = set(chart.get_onsets())
    y_true, y_scores_pkalgn = align_onsets_to_sklearn(true_onsets, pred_onsets, scores, tolerance=args.eval_align_tolerance)
    y_scores = scores[chart.get_first_onset():chart.get_last_onset() + 1]
    y_xentropies = xentropies[chart.get_first_onset():chart.get_last_onset() + 1]

    return y_true, y_scores, y_xentropies, y_scores_pkalgn

def eval_metrics_for_scores(y_true, y_scores, y_xentropies, y_scores_pkalgn, given_threshold_names=[], given_thresholds=[], return_prc=False):
    nonsets = np.sum(y_true)
    if y_xentropies is not None:
        xentropy_avg = np.mean(y_xentropies)
        pos_xentropy_avg = np.sum(np.multiply(y_xentropies, y_true)) / nonsets
    else:
        xentropy_avg = 0.
        pos_xentropy_avg = 0.

    # calculate ROC curve
    fprs, tprs, thresholds = roc_curve(y_true, y_scores_pkalgn)
    auroc = auc(fprs, tprs)

    # calculate PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores_pkalgn)
    # https://github.com/scikit-learn/scikit-learn/issues/1423
    auprc = auc(recalls, precisions)

    # find best fscore and associated values
    fscores_denom = precisions + recalls
    fscores_denom[np.where(fscores_denom == 0.0)] = 1.0
    fscores = (2 * (precisions * recalls)) / fscores_denom
    fscore_max_idx = np.argmax(fscores)
    precision, recall, fscore, threshold_ideal = precisions[fscore_max_idx], recalls[fscore_max_idx], fscores[fscore_max_idx], thresholds[fscore_max_idx]

    # calculate density
    predicted_steps = np.where(y_scores_pkalgn >= threshold_ideal)
    density_rel = float(len(predicted_steps[0])) / float(nonsets)

    # calculate accuracy
    y_labels = np.zeros(y_scores_pkalgn.shape[0], dtype=int)
    y_labels[predicted_steps] = 1
    accuracy = accuracy_score(y_true.astype(int), y_labels)

    # calculate metrics for fixed thresholds
    metrics = {}
    for threshold_name, threshold in zip(given_threshold_names, given_thresholds):
        threshold_closest_idx = np.argmax(thresholds >= threshold)
        precision_closest, recall_closest, fscore_closest, threshold_closest = precisions[threshold_closest_idx], recalls[threshold_closest_idx], fscores[threshold_closest_idx], thresholds[threshold_closest_idx]
        metrics['precision_{}'.format(threshold_name)] = precision_closest
        metrics['recall_{}'.format(threshold_name)] = recall_closest
        metrics['fscore_{}'.format(threshold_name)] = fscore_closest
        metrics['threshold_{}'.format(threshold_name)] = threshold_closest

    # aggregate metrics
    metrics['xentropy_avg'] = xentropy_avg
    metrics['pos_xentropy_avg'] = pos_xentropy_avg
    metrics['auroc'] = auroc
    metrics['auprc'] = auprc
    metrics['fscore'] = fscore
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['threshold'] = threshold_ideal
    metrics['accuracy'] = accuracy
    metrics['perplexity'] = np.exp(xentropy_avg)
    metrics['density_rel'] = density_rel

    if return_prc:
        return metrics, (precisions, recalls)
    else:
        return metrics

if __name__ == '__main__':
    args = parse_args()
    train_onset_model(args)
