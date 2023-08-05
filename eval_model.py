from architecture import Model1
from util import log, Config
import torch
import os
import numpy
from data_loader import make_evaluation_loader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score


def eval_model(model, testset_path):

    def plot_confusion_matrix(conf_matrix, title):
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        classes = ['Masked', 'Unmasked']
        tick_marks = numpy.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    test_loader = make_evaluation_loader(testset_path)

    def get_class(idx):
        return test_loader.dataset.classes[idx]

    # run model on test set, collect predictions and actual labels
    labels_true = []
    labels_predictions = []
    total = len(test_loader.dataset)
    for data, label in test_loader:
        actual = label.item()
        labels_true.append(actual.numpy())
        prediction = 1 if torch.sigmoid(model(data)).item() > 0.5 else 0
        labels_predictions.append(prediction)

    labels_true = numpy.array(labels_true)
    labels_predictions = numpy.array(labels_predictions)

    # calculate metrics
    precision = precision_score(labels_true, labels_predictions)
    recall = recall_score(labels_true, labels_predictions)
    accuracy = accuracy_score(labels_true, labels_predictions)
    f1 = f1_score(labels_true, labels_predictions)

    # create confusion matrix
    conf_matrix = confusion_matrix(labels_true, labels_predictions)

    # plot confusion matrix
    plot_confusion_matrix(conf_matrix, title='Model Confusion Matrix')
    plt.savefig(f'{model.__class__.__name__}confusion_matrix.png')

    # print metrics
    print(
        f"Model: {model.__class__.__name__}  - Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, F1 Score: {f1}")


if __name__ == "__main__":
    model = Model1()
    config = Config()

    # load pre-trained model
    model_path = os.path.abspath(
        os.getenv("MODEL_PATH") or config.get(
            "Paths", "model") or "./trained.pt"
    )
    log.info(f"Model path: {model_path}")

    # load path to testing set
    testset_path = os.path.abspath(
        os.getenv("TESTSET_PATH")
        or config.get("Paths", "test_set")
        or "./datasets/testing_set"
    )
    log.info(f"Test-set path: {testset_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    eval_model(model, testset_path)
