import csv
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_keras_history import plot_history
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, roc_auc_score, average_precision_score, \
    auc, precision_recall_curve, accuracy_score, balanced_accuracy_score, top_k_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
from tensorflow import keras
from keras.utils import plot_model

from cnn_transformer import compile_cnn_transformer, FeatureExtractorSpec, scheduler, \
    compile_time_distributed_model, compile_time_distributed_rnn_model
from vivit import compile_vivit_model


def get_classification_table(y_true, y_score):
    classes = list(set(y_true))
    class_table = [[0] * (len(classes) + 1) for _ in range(len(classes))]
    for y, y_s in zip(y_true, y_score):
        class_table[int(y)][0] += 1
        class_table[int(y)][y_s + 1] += 1
    rows = [f"Class {i}" for i in classes]
    cols = [" "] + rows
    return pd.DataFrame(class_table, index=rows, columns=cols)


def calculate_auprc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def calculate_tpr_10_per(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_at_1_fpr = tpr[np.abs(fpr - 0.1).argmin()]
    return tpr_at_1_fpr


def calculate_tpr_5_per(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_at_5_fpr = tpr[np.abs(fpr - 0.05).argmin()]
    return tpr_at_5_fpr


def calculate_tpr_1_per(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_at_5_fpr = tpr[np.abs(fpr - 0.01).argmin()]
    return tpr_at_5_fpr


def calculate_eer(y_true, y_score):
    """
    Returns the equal error rate for a binary classifier output.
    https://github.com/scikit-learn/scikit-learn/issues/15247#issuecomment-542138349
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


full_frame_0_0_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.DENSENET,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_frame_0_1_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.XCEPTION,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_frame_0_2_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.INCEPTIONRESNETV2,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_frame_0_3_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.NASNETLARGE,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_frame_0_4_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.EFFICIENTNETV2L,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_frame_0_5_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.CONVNEXTXLARGE,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_frame_0_6_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.RESNET152V2,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_frame_0_7_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.VGG16,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_0_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.DENSENET,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_1_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.XCEPTION,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_2_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.INCEPTIONRESNETV2,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_3_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.NASNETLARGE,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_4_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.EFFICIENTNETV2L,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_5_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.CONVNEXTXLARGE,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_6_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.RESNET152V2,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_0_7_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.VGG16,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

tracklet_3_0_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.DENSENET,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_3_1_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.XCEPTION,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_3_2_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.INCEPTIONRESNETV2,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_3_3_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.NASNETLARGE,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_3_4_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.EFFICIENTNETV2L,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_3_5_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.CONVNEXTXLARGE,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_3_6_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.RESNET152V2,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_3_7_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.VGG16,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_1_8_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.NONE,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 1
    })

full_frame_1_8_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.NONE,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 1
    })

full_frame_3_0_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.DENSENET,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

full_frame_3_1_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.XCEPTION,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

full_frame_3_2_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.INCEPTIONRESNETV2,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

full_frame_3_3_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.NASNETLARGE,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

full_frame_3_4_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.EFFICIENTNETV2L,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

full_frame_3_5_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.CONVNEXTXLARGE,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

full_frame_3_6_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.RESNET152V2,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

full_frame_3_7_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.VGG16,
        "tracklet": False,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

binary_label_map = ['hitting']
binary_full_label_map = ['no pbx', 'hitting']

# mc_label_map = ['hitting', 'flop', 'biting', 'kick_hit object', 'throwing object', 'pushing', 'grab_scratch']
# mc_full_label_map = ['no pbx', 'hitting', 'flop', 'biting', 'kick_hit object', 'throwing object', 'pushing',
#                      'grab_scratch']

mc_label_map = ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting',
                 'choking',
                 'sib-head bang', 'sib-head hit', 'sib-self-hit', 'sib-biting', 'sib-eye poke', 'sib-body slam',
                 'sib-hair pull', 'sib-choking', 'sib-pinch_scratch', 'throwing object', 'kick_hit object',
                 'flip furniture',
                 'flop']

mc_full_label_map = ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting',
                     'choking',
                     'sib-head bang', 'sib-head hit', 'sib-self-hit', 'sib-biting', 'sib-eye poke', 'sib-body slam',
                     'sib-hair pull', 'sib-choking', 'sib-pinch_scratch', 'throwing object', 'kick_hit object',
                     'flip furniture',
                     'flop', 'no pbx']

nas_experiment_schedule = [
    tracklet_0_0_experiment,
    full_frame_0_0_experiment
]
preliminary_experiment_schedule = [
    tracklet_0_0_experiment,
    tracklet_0_1_experiment,
    tracklet_0_2_experiment,
    tracklet_0_3_experiment,
    tracklet_0_4_experiment,
    tracklet_0_5_experiment,
    tracklet_0_6_experiment,
    tracklet_0_7_experiment,
    tracklet_3_0_experiment,
    tracklet_3_1_experiment,
    tracklet_3_2_experiment,
    tracklet_3_3_experiment,
    tracklet_3_4_experiment,
    tracklet_3_5_experiment,
    tracklet_3_6_experiment,
    tracklet_3_7_experiment,
    full_frame_0_0_experiment,
    full_frame_0_1_experiment,
    full_frame_0_2_experiment,
    full_frame_0_3_experiment,
    full_frame_0_4_experiment,
    full_frame_0_5_experiment,
    full_frame_0_6_experiment,
    full_frame_0_7_experiment,
    full_frame_3_0_experiment,
    full_frame_3_1_experiment,
    full_frame_3_2_experiment,
    full_frame_3_3_experiment,
    full_frame_3_4_experiment,
    full_frame_3_5_experiment,
    full_frame_3_6_experiment,
    full_frame_3_7_experiment,
    full_frame_1_8_experiment,
    tracklet_1_8_experiment,
]

tracklet_2_6_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.RESNET152V2,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 2
    })

tracklet_2_7_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.VGG16,
        "tracklet": True,
        "label-map": ['hitting'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 2
    })

tracklet_0_6_agg_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.RESNET152V2,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                      'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                           'biting', 'choking'],
        "model_choice": 0
    })

tracklet_0_7_agg_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.VGG16,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                      'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                           'biting', 'choking'],
        "model_choice": 0
    })

tracklet_3_6_agg_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.RESNET152V2,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                      'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                           'biting', 'choking'],
        "model_choice": 3
    })

tracklet_3_7_agg_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.VGG16,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                      'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                           'biting', 'choking'],
        "model_choice": 3
    })

tracklet_3_9_agg_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.HOG,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                      'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 3
    })

tracklet_0_9_agg_experiment = dict(
    {
        "dense-net": FeatureExtractorSpec.HOG,
        "tracklet": True,
        "label-map": ['hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling',
                      'biting', 'choking'],
        "full-label-map": ['no pbx', 'hitting'],
        "model_choice": 0
    })

full_experiment_schedule = [
    # tracklet_0_0_experiment,
    # tracklet_0_1_experiment,
    tracklet_0_5_experiment,
    # tracklet_0_6_experiment,
    # tracklet_0_7_experiment,
    # tracklet_0_9_agg_experiment,
    # tracklet_3_0_experiment,
    # tracklet_3_1_experiment,
    tracklet_3_5_experiment,
    # tracklet_3_6_experiment,
    # tracklet_3_7_experiment,
    # tracklet_3_9_agg_experiment,
]


def run_experiment(train_data, train_labels, test_data, test_labels,
                   classes,
                   output_dir, epochs, image_size, seq_len, num_features, checkpoint_dir, fold,
                   model_choice, batch_size,
                   label_map,
                   val_x, val_y
                   ):
    ck_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_dir + '/' + str(fold) + '_weights.{epoch:02d}-{val_accuracy:.4f}.hdf5' if classes == 1
        else checkpoint_dir + '/' + str(fold) + '_weights.{epoch:02d}-{val_top_k_categorical_accuracy:.4f}.hdf5',
        monitor='val_accuracy' if classes == 1 else 'val_top_k_categorical_accuracy',
        mode='max', verbose=2,
        save_best_only=True,
        save_weights_only=True
    )
    # Compile and train model
    if model_choice == 0:
        model = compile_cnn_transformer(seq_len, num_features, classes)
    elif model_choice == 1:
        model = compile_vivit_model(input_shape=(seq_len, image_size, image_size, 1), num_classes=classes)
    elif model_choice == 2:
        model = compile_time_distributed_rnn_model(seq_len, num_features, batch_size, classes)
    elif model_choice == 3:
        model = compile_time_distributed_model(seq_len, num_features, batch_size, classes)
    else:
        raise ValueError()

    plot_model(model, to_file=output_dir + '/model_plot.png', show_shapes=True, show_layer_names=True)

    history = model.fit(
        train_data,
        train_labels,
        batch_size=batch_size,
        validation_data=(test_data, test_labels),
        epochs=epochs,
        callbacks=[
            ck_callback,
        ]
    )
    # Create the history directory and save history as CSV
    hist_df = pd.DataFrame(history.history)
    history_dir = os.path.join(output_dir, f'history')
    if not os.path.exists(history_dir):
        os.mkdir(history_dir)
    hist_csv_file = os.path.join(history_dir, 'history.csv')
    with open(hist_csv_file, mode='w', newline='') as f:
        hist_df.to_csv(f)
    # Create figures directory and save history figure
    figures_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    plot_history(history, path=f"{figures_dir}/interpolated_{fold}_{epochs}_{image_size}.png", interpolate=True)
    plt.close()
    del history
    # Get latest weights and load into model
    weights_dir = pathlib.Path(checkpoint_dir)
    weights_pattern = r'*.hdf5'
    latest_weight = max(weights_dir.glob(weights_pattern), key=lambda f: f.stat().st_ctime)
    print(f"Loaded weights for testing: {latest_weight}")
    conv_epoch = int(str(latest_weight).split('.')[1].split('-')[0])

    # Test model using test data
    model.load_weights(latest_weight)
    y_true = val_y
    y_probs = model.predict(val_x)
    y_pred = y_probs.reshape(-1).tolist()
    class_results = []
    # results_csv_file = os.path.join(history_dir, str(len(next(os.walk(history_dir)))) + 'results.csv')
    pred_csv_file = os.path.join(history_dir, f'{fold}_pred.csv')

    # Compute metrics
    if classes == 1:
        results_headers = ['Class', 'Test Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC-AUC', 'mAP', 'EER',
                           'TPR@1%', 'TPR@5%', 'TPR@10%', 'AUPRC', 'Kappa']
        pd.DataFrame([y_true, y_pred], index=['True', 'Predicted']).to_csv(pred_csv_file)
        y_score = y_probs > 0.5
        accuracy = balanced_accuracy_score(y_true, y_score)
        kappa = cohen_kappa_score(y_true, y_score)
        f1_s = f1_score(y_true, y_score, average='macro', zero_division=0)
        recall = recall_score(y_true, y_score, average='macro', zero_division=0)
        precision = precision_score(y_true, y_score, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
        mean_average_precision = average_precision_score(y_true, y_pred)
        eer = calculate_eer(y_true, y_pred)
        tpr_10_per = calculate_tpr_10_per(y_true, y_pred)
        tpr_5_per = calculate_tpr_5_per(y_true, y_pred)
        tpr_1_per = calculate_tpr_1_per(y_true, y_pred)
        pr_auc = calculate_auprc(y_true, y_pred)
        class_results.append([label_map[1], accuracy, f1_s, recall, precision, roc_auc,
                              mean_average_precision, eer,
                              tpr_1_per, tpr_5_per, tpr_10_per, pr_auc, kappa])
    else:
        results_headers = ['Class', 'Top-1', 'Top-3', 'F1 Score', 'Recall', 'Precision', 'ROC-AUC', 'mAP', 'EER',
                           'TPR@1%', 'TPR@5%', 'TPR@10%', 'AUPRC', 'Kappa']
        y_hat = label_binarize(y_probs.argmax(axis=-1), classes=list(np.arange(classes + 1)))
        y_score = label_binarize(y_true, classes=list(np.arange(classes + 1)))
        pd.DataFrame([y_true.argmax(axis=-1), y_probs.argmax(axis=-1)], index=['True', 'Predicted']).to_csv(
            pred_csv_file)
        top5 = top_k_accuracy_score(y_true.argmax(axis=-1), y_probs, k=3)
        for i in range(classes + 1):
            accuracy = balanced_accuracy_score(y_score[:, i], y_hat[:, i])
            kappa = cohen_kappa_score(y_score[:, i], y_hat[:, i])
            recall = recall_score(y_score[:, i], y_hat[:, i], average='macro', zero_division=0)
            precision = precision_score(y_score[:, i], y_hat[:, i], average='macro', zero_division=0)
            f1_s = f1_score(y_score[:, i], y_hat[:, i], average='weighted')
            tpr_10_per = calculate_tpr_10_per(y_score[:, i], y_probs[:, i])
            tpr_5_per = calculate_tpr_5_per(y_score[:, i], y_probs[:, i])
            tpr_1_per = calculate_tpr_1_per(y_score[:, i], y_probs[:, i])
            eer = calculate_eer(y_score[:, i], y_probs[:, i])
            mean_average_precision = average_precision_score(y_score[:, i], y_probs[:, i], average='weighted')
            roc_auc = roc_auc_score(y_score[:, i], y_probs[:, i], average='weighted')
            pr_auc = calculate_auprc(y_score[:, i], y_probs[:, i])
            class_results.append([label_map[i], accuracy, top5, f1_s, recall, precision, roc_auc,
                                  mean_average_precision, eer,
                                  tpr_1_per, tpr_5_per, tpr_10_per, pr_auc, kappa])
    # Write metrics to file
    # with open(results_csv_file, mode='w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(results_headers)
    #     for cr in class_results:
    #         writer.writerow(cr)
    return model, class_results, conv_epoch
