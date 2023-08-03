# dl_facemask_detector

A simple face mask detector using deep learning.

## Table of Contents

1. [Dataset](#dataset)
2. [Model](#model)
3. [Setup](#setup)
4. [Demo](#demo)
5. [Running your own training loop](#training)
6. [Running your own evaluation loop](#classification)
7. [Report](#report)

## Dataset

For this project we have used GitHub user **X-zhangyang**'s
"Real-World-Masked-Face-Dataset". The original source can be found
[here](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset).

For the sake of convenience, we have already pre-processed the dataset and are
mirroring it
[here](https://drive.google.com/file/d/1ip04I_bX-PuIXnuzoEhAL_UZW1qTXu8y/view?usp=sharing).


## Model

The model architecture is very loosely based on
[LeNet-5](http://yann.lecun.com/exdb/lenet/), with some added layers and
increased complexity. The model definition can be found in
[architecture.py](./architecture.py).

More details regarding the architecture and how it compares to other
state of the art solutions can be found in the [project report](#report).

A link to a pre-trained model can be found
[here](https://drive.google.com/file/d/14Vk8ochj48OGOw6KAUaD4nEakPPcgIo4/view?usp=sharing).

## Setup

1. Configure your local environment with the necessary dependencies. We
   recommend using [conda](https://docs.conda.io/en/latest/) for setting up
   your environment. This project uses Python `3.10`.

    ```shell
    conda create -n <env_name> python=3.10
    conda activate <env_name>
    pip install -r requirements.txt
    ```

**⚠️ If you just want to run the [demo notebook](#demo), you can skip the next
steps entirely.**

2. Copy [`example_config.ini`](./example_config.ini) to `config.ini`. Make sure
   to adjust the config values based on your setup. Alternatively, you may also
   specify the values by exporting the following environment variables (see the
   file [`.envrc`](./.envrc) for an example):

   | Value          | Environment Var |
   |--------------- | --------------- |
   | Dataset path   | `$DATASET_PATH`   |
   | Testset path   | `$TESTSET_PATH`   |
   | Model path     | `$MODEL_PATH`        |


3. (**Optional**): Download the pre-trained model and dataset. These can each
   be downloaded by running `make model` and `make dataset` respectively. The
   output path(s) can be overwritten by specifying `$OUT`.

## Demo

We have included a [Jupyter notebook](./demo.ipynb) in this project as a means
of demonstrating the model's performance. The demo can be run **as-is**
(provided you have completed the project [setup](#setup)). Utility functions to
download the necessary testsets and pre-trained model weights are provided in
notebook.

The demo notebook can be started by running `jupyter-lab --port 8080` and then
opening the file [`demo.ipynb`](./demo.ipynb) from within the web-GUI.

We have also included a live-evaluation script which will use to model to
classify the existence of a masked individual from a webcam feed (or lack
thereof). The script is a little unreliable, so to get the "best" picture of
the model we recommend running the notebook first (see also: evaluation of the
dataset in the [report](#report)). To run the webcam-evaluator, ensure you have
followed the [setup](#setup) guide and then execute the following commands:

```shell
# If you haven't already, download the model
OUT=$(pwd)/model.pt make model

MODEL=$(pwd)/model.pt python3 ./eval_model_webcam.py
```

The webcam feed is coded to appear grayscale when no mask is detected and
colored otherwise.

## Training

To run a training loop, ensure you have taken the following steps:

1. Complete the project [setup](#setup).
2. If you haven't already, download the dataset:

    ```shell
    OUT=$(pwd)/dataset make dataset
    ```

The training script can then be run with the following command:

```shell
WANDB_MODE=disabled DATASET_PATH=$(pwd)/dataset/train python3 train_model.py
```

## Evaluation

To run an evaluation loop on a batch of images, ensure you have taken the
following steps:

1. Complete the project [setup](#setup).
2. If you haven't already, download the dataset and model:

    ```shell
    OUT=$(pwd)/dataset make dataset
    OUT=$(pwd)/model.pt make model
    ```

The evaluation script can then be run with the following command:

```shell
MODEL=$(pwd)/model.pt TESTSET_PATH=$(pwd)/dataset/test python3 eval_model.py
```

<!-- TODO: should we still support this? -->
<!-- To classify a single image, you may also invoke the `single` subcommand. You -->
<!-- will need to provide your own image. We recommend a square image in which the -->
<!-- subject's face is relatively centered for the best performance. -->
<!---->
<!-- ```shell -->
<!-- MODEL=$(pwd)/model.pt python3 eval_model.py single --in=/path/to/my/image.png -->
<!-- ``` -->

## Report

We have summarized our findings in a project report. You can view the rendered
document [here](./report/report.pdf).

To build the report yourself, run `make report` (requires tex to be configured
on your system).

