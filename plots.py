####################################################
#
#   Authors: Adriana Galán & Marta Goyena
#   Project: MusicTaste - Genre Classification
#   Year: 2024
#
####################################################


# Libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import seaborn as sns


# Uses categorical variables because it is what is used in the deep learning (DL) models
def accloss(history, modelname, savepath, savename):
    # 1. Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{modelname} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # Save the figure
    fig_save_path = f'{savepath}/accuracy_{savename}.png'
    plt.savefig(fig_save_path)

    # Show figure
    plt.show()

    # 2. Plot loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{modelname} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    fig_save_path = f'{savepath}/loss_{savename}.png'
    plt.savefig(fig_save_path)
    plt.show()

    return


def cm(real_labels, pred_labels, modelname, classes, savepath, savename):
    cm = confusion_matrix(real_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=45)
    plt.title(f'{modelname} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_save_path = f'{savepath}/cm_{savename}.png'
    plt.savefig(fig_save_path)
    plt.show()

    return


def cm_mutiabel(real_labels, pred_labels, modelname, classes, savepath, savename):
# Plot confusion matrix
    confusion = multilabel_confusion_matrix(real_labels, pred_labels)
    fig = plt.figure(figsize = (14, 8))
    for i, (label, matrix) in enumerate(zip(real_labels, confusion)):
        plt.subplot(f'23{i+1}')
        labels = [f'not_{label}', label]
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(labels[0])
    plt.xticks(rotation=45)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_save_path = f'{savepath}/cm_{savename}.png'
    plt.savefig(fig_save_path)
    plt.show()
    plt.tight_layout()