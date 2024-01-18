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
def accloss(history, model_name, save_path, save_name):
    # 1. Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy']) #categorical_accuracy
    plt.plot(history.history['val_accuracy']) #val_categorical_accuracy
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # Save the figure
    fig_save_path = f'{save_path}/accuracy_{save_name}.png'
    plt.savefig(fig_save_path)

    # Show figure
    plt.show()

    # 2. Plot loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    fig_save_path = f'{save_path}/loss_{save_name}.png'
    plt.savefig(fig_save_path)
    plt.show()

    return


def cm(real_labels, pred_labels, model_name, classes, save_path, save_name):
    cm = confusion_matrix(real_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=45)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_save_path = f'{save_path}/cm_{save_name}.png'
    plt.savefig(fig_save_path)
    plt.show()

    return


def cm_mutilabel(y_true, y_pred, model_name, classes, save_path, save_name):
    # Confusion matrices
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    # Iteration to print all of them
    for i, matrix in enumerate(confusion_matrices):
        print(f"Matriz de Confusión para la Clase {classes[i]}:\n", matrix)
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Multilabel Confusion Matrix for {classes[i]} Genre with {model_name}')
        plt.xticks(rotation=45)
        fig_save_path = f'{save_path}/cm_{save_name}_{classes[i]}.png'
        plt.savefig(fig_save_path)
        plt.show()

    return
