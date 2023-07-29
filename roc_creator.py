import glob
import os

import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_confusion_matrix(cm,
                          target_names,
                          output_path,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names, rotation=90, va="center")

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}%".format(cm[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}%".format(cm[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\nAccuracy={:0.2f}%; Misclass={:0.2f}%'.format(accuracy*100, misclass*100))
    plt.savefig(output_path)
    plt.clf()


data_files = [
    r".\MLHC\All\Tracklet_0_0_1_hitting_None\history\4_pred.csv",
    r".\MLHC\All\Tracklet_0_1_1_hitting_None\history\2_pred.csv",
    r".\MLHC\All\Tracklet_0_5_1_hitting_None\history\1_pred.csv",
    r".\MLHC\All\Tracklet_0_6_1_hitting_None\history\2_pred.csv",
    r".\MLHC\All\Tracklet_0_7_1_hitting_None\history\1_pred.csv",
    r".\MLHC\All\Tracklet_0_9_1_hitting_None\history\1_pred.csv"
]

fes = [
    'DenseNet',
    'Xception',
    'ConvNeXtXL',
    'ResNet152V2',
    'VGG16',
    'HOG'
]

for file, fe in zip(data_files, fes):
    pd_file = pd.read_csv(file)
    pd_file = pd_file.set_index('Unnamed: 0')
    pd_file = pd_file.transpose()
    cm = confusion_matrix(pd_file['True'], pd_file['Predicted'] > 0.5)
    plot_name = f"Confusion Matrix for {fe} Transformer Model"
    # plot_confusion_matrix(cm, ["Other Behaviors", "Aggression"], plot_name + ".png",
    #                       title=plot_name)
    fpr, tpr, thresh = roc_curve(pd_file['True'], pd_file['Predicted'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{fe} AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Receiver Operating Characteristic for Transformer Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig("ROC_Transformer.png")
plt.clf()

data_files = [
    r".\MLHC\All\Tracklet_3_0_1_hitting_None\history\3_pred.csv",
    r".\MLHC\All\Tracklet_3_1_1_hitting_None\history\1_pred.csv",
    r".\MLHC\All\Tracklet_3_5_1_hitting_None\history\1_pred.csv",
    r".\MLHC\All\Tracklet_3_6_1_hitting_None\history\3_pred.csv",
    r".\MLHC\All\Tracklet_3_7_1_hitting_None\history\4_pred.csv",
    r".\MLHC\All\Tracklet_3_9_1_hitting_None\history\1_pred.csv"
]

for file, fe in zip(data_files, fes):
    pd_file = pd.read_csv(file)
    pd_file = pd_file.set_index('Unnamed: 0')
    pd_file = pd_file.transpose()
    cm = confusion_matrix(pd_file['True'], pd_file['Predicted'] > 0.5)
    plot_name = f"Confusion Matrix for {fe} Recurrent Model"
    # plot_confusion_matrix(cm, ["Other Behaviors", "Aggression"], plot_name + ".png",
    #                       title=plot_name)
    fpr, tpr, thresh = roc_curve(pd_file['True'], pd_file['Predicted'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{fe} AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Receiver Operating Characteristic for Recurrent Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig("ROC_GRU.png")
plt.clf()

subjects = [
    'Patient 1',
    'Patient 3',
    'Patient 5',
    'Patient 8'
]

data_files = [
    r".\MLHC\Patients\Tracklet_0_5_1_hitting_p001\history\4_pred.csv",
    r".\MLHC\Patients\Tracklet_0_5_1_hitting_p003\history\5_pred.csv",
    r".\MLHC\Patients\Tracklet_0_5_1_hitting_p005\history\1_pred.csv",
    r".\MLHC\Patients\Tracklet_0_5_1_hitting_p008\history\5_pred.csv",
]

for file, fe in zip(data_files, subjects):
    pd_file = pd.read_csv(file)
    pd_file = pd_file.set_index('Unnamed: 0')
    pd_file = pd_file.transpose()
    cm = confusion_matrix(pd_file['True'], pd_file['Predicted'] > 0.5)
    plot_name = f"Confusion Matrix for {fe} Transformer Model"
    # plot_confusion_matrix(cm, ["Other Behaviors", "Aggression"], plot_name + ".png",
    #                       title=plot_name)
    fpr, tpr, thresh = roc_curve(pd_file['True'], pd_file['Predicted'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{fe} AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Receiver Operating Characteristic for Transformer Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig("Patients_ROC_Transformer.png")
plt.clf()

data_files = [
    r".\MLHC\Patients\Tracklet_3_5_1_hitting_p001\history\5_pred.csv",
    r".\MLHC\Patients\Tracklet_3_5_1_hitting_p003\history\3_pred.csv",
    r".\MLHC\Patients\Tracklet_3_5_1_hitting_p005\history\4_pred.csv",
    r".\MLHC\Patients\Tracklet_3_5_1_hitting_p008\history\2_pred.csv",
]

for file, fe in zip(data_files, subjects):
    pd_file = pd.read_csv(file)
    pd_file = pd_file.set_index('Unnamed: 0')
    pd_file = pd_file.transpose()
    cm = confusion_matrix(pd_file['True'], pd_file['Predicted'] > 0.5)
    plot_name = f"Confusion Matrix for {fe} Recurrent Model"
    # plot_confusion_matrix(cm, ["Other Behaviors", "Aggression"], plot_name + ".png",
    #                       title=plot_name)
    fpr, tpr, thresh = roc_curve(pd_file['True'], pd_file['Predicted'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{fe} AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Receiver Operating Characteristic for Recurrent Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig("Patients_ROC_GRU.png")
plt.clf()
