from architecture import Model1
from config import Config
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import logging as log
import os

log.basicConfig(level=log.INFO, format="[%(levelname)s] %(message)s")


def eval_model(model, test_set):
    def get_class(idx):
        return test_set.classes[idx]

    log.info(f"test dataset: {len(test_set)} samples")
    log.info(f"Detected classes: {test_set.class_to_idx}")
    test_loader = DataLoader(test_set, shuffle=True)

    num_correct = 0
    total = len(test_set)
    for data, label in test_loader:
        actual = label.item()
        prediction = 1 if model(data).item() > 0.5 else 0
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
        os.getenv("MODEL") or config.get("Paths", "model") or "./trained.pt"
    )

    # load path to testing set
    testset_path = os.path.abspath(
        os.getenv("TESTSET_PATH")
        or config.get("Paths", "test_set")
        or "./datasets/testing_set"
    )
    log.info(f"Test-set path: {testset_path}")

    test_set = ImageFolder(
        root=testset_path,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    eval_model(model, test_set)
