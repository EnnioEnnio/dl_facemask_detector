# dl_facemask_detector

A simple face mask detector using deep learning.

## Dataset
We have used the Real-World-Masked-Face-Dataset
 from [here](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) for training the model.

To download the dataset, you can run `make dataset` (requires environment
[setup](#setup))

### Data Processing

If you wish to manually preprocess the dataset, you can execute
`./data_loader.py` as a standalone script. This will take care of normalizing
image dimensions and introducing rotational variations to the dataset.

```shell
# example for processing masked dataset
python3 ./data_loader.py \
    -i ../datasets-raw/AFDB_masked_face_dataset \
    -o ./processed-masked --flatten \
    --rotate \
    -v \
    --size 256 256
```

Be aware that you will have to use `--no-rotate` if you do not want the images to be rotated.

## Setup
we recommend using conda for setting up the environment. 

```shell
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

## Model
[Placeholder]

## Training
Rename the `example_config.ini` to `config.ini` and change the path to your preprocessed data. 
In `train_model.py` select the architecture from `architecture.py` you want to use and run 

```shell 
python train_model.py
```


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



