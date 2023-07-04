# dl_facemask_detector

A simple face mask detector using deep learning.

## Dataset
We have used the Real-World-Masked-Face-Dataset
 from [here](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) for training the model.

## Setup
we recommend using conda for setting up the environment. 

```shell
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

## Model
[Placeholder]


## Usage
### Evaluation
[Placeholder]

To evaluate the model on the test data, run the following command:

```shell 
python eval_model.py --model_path <path_to_model> --data_path <path_to_test_data>
```

### Classification
[Placeholder]

To classify an image, run the following command:

```shell 
python run_model.py --model_path <path_to_model> --image_path <path_to_image>
```



