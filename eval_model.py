from architecture import Model1
from util import log, Config
import torch
import os
from data_loader import make_evaluation_loader


def eval_model(model, testset_path):
    test_loader = make_evaluation_loader(testset_path)

    def get_class(idx):
        return test_loader.dataset.classes[idx]

    num_correct = 0
    total = len(test_loader.dataset)
    for data, label in test_loader:
        actual = label.item()
        prediction = 1 if torch.sigmoid(model(data)).item() > 0.5 else 0
        print(
            f"Actual: {actual} ({get_class(actual)}), Prediction: {prediction} ({get_class(prediction)})"
        )
        if actual == prediction:
            num_correct += 1
    print(f"Accuracy: {num_correct/total}")


if __name__ == "__main__":
    model = Model1()
    config = Config()

    # load pre-trained model
    model_path = os.path.abspath(
        os.getenv("MODEL_PATH") or config.get("Paths", "model") or "./trained.pt"
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
