# EfficientDet (fork from https://github.com/xuannianz/EfficientDet)
This is an implementation of [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) for object detection on Keras and Tensorflow. 
The project is based on the official implementation [google/automl](https://github.com/google/automl), [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
and the [qubvel/efficientnet](https://github.com/qubvel/efficientnet). 

## The project observed the following structure
```
.
├── EfficientDet (repo)
├── EfficientDet_Notebook.ipynb
├── Local_Path (datasets)
│   ├── Images
│   │   └── img.png
│   └── models
│       └── dataset1
|           ├── classes.csv
|           ├── train_val.csv
|           └── test.csv
├── LOCAL_ROOT_PATH (path of an experiment)                    
│   ├── classes.csv
│   ├── train_val.csv
│   ├── test.csv
|   |── logs
|   |   ├── config.yaml (saved config)
|   |   ├── stdout.txt (training logs)
|   |   └── events.out.tfevents.XXXX.jupyter (tensorboard logs)
|   └── snapshots
|       └── 0_0.h5 (snapshot saved)
└── ...
```

## How it works - Just use the following function in your jupyter notebook
Warning : training_utils being a module of this repo you need to be at the root_path or use sys.path.insert(0, root_path).  
Dependencies are written in requirements.txt, you can also use py-air-efficientdet kernel.  

```python  
import sys
sys.path.insert(0, '/home/c3/jupyter_root_dir/Mathieu/EfficientDet')

# confirm TensorFlow sees the GPU
assert 'GPU' in str(device_lib.list_local_devices())

# confirm TensorFlow is built with cuda
assert tf.test.is_built_with_cuda()

# confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))

# confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
assert len(keras.backend.tensorflow_backend._get_available_gpus()) > 0

REMOTE_EXPERIMENT_PATH = 'azure://predev-shellair/fs/air/prime/c3_built_datasets/results/test_backup' # CHANGE THIS ACCORDINGLY
# Local Paths
LOCAL_PATH = '/home/c3/jupyter_root_dir/data/detection/' # CHANGE THIS ACCORDINGLY
LOCAL_DATASETS_PATH = os.path.join(LOCAL_PATH, 'models')

def train(config):
    
    from training_utils import create_callbacks, efficientdet_training

    # Local paths
    LOCAL_CLASSES_PATH = f'{LOCAL_ROOT_PATH}/classes.csv'
    LOCAL_ANNOTATIONS_PATH = f'{LOCAL_ROOT_PATH}/train_val.csv'
    LOCAL_VALIDATIONS_PATH = f'{LOCAL_ROOT_PATH}/test.csv'
    LOCAL_SNAPSHOTS_PATH = f'{LOCAL_ROOT_PATH}/snapshots'
    LOCAL_LOGS_PATH = f'{LOCAL_ROOT_PATH}/logs'

    # copy dataset metadata files
    DATASETS_PATH = f"{LOCAL_DATASETS_PATH}/{config['dataset_name']}"
    os.makedirs(LOCAL_ROOT_PATH, exist_ok=True)
    for filename in ('test.csv','test2.csv','train_val.csv','classes.csv','classes_count.csv'):
        source_file_path = f'{DATASETS_PATH}/{filename}'
        target_file_path = f'{LOCAL_ROOT_PATH}/{filename}'    
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, target_file_path)
            # TODO TO REMOVE
        else:
            print(
                f"WARNING: Source file path {source_file_path} does not exist!"
            )
            
    print(LOCAL_ROOT_PATH)
    model = efficientdet_training(
        config,
        LOCAL_ANNOTATIONS_PATH,
        LOCAL_ROOT_PATH,
        LOCAL_CLASSES_PATH,
        LOCAL_VALIDATIONS_PATH,
        LOCAL_LOGS_PATH,
        LOCAL_SNAPSHOTS_PATH
    )

    c3.Client.uploadLocalClientFiles(localPath=LOCAL_ROOT_PATH, dstUrlOrEncodedPath=REMOTE_EXPERIMENT_PATH)
    print('End of training.')
```     
      
LOCAL_ANNOTATIONS_PATH being the path of the train annotations (following the structure from the original repo).   
LOCAL_ROOT_PATH being the directory path of your experiment.   
LOCAL_CLASSES_PATH being the path of your description classes (following the structure from the original repo).   
LOCAL_VALIDATIONS_PATH being the path of the val annotations.   
LOCAL_LOGS_PATH being the directory path where to logs everything useful.   
LOCAL_SNAPSHOTS_PATH being the directory path where to save model snapshots.   

## An example of a valid config used by the method efficientdet_training
```YAML
architecture: efficientdet # field used as comment
experiment_name: augmented_pool3_phi3_focal_1 # field used as comment
dataset_name: pool3 # used to retrieve corresponding dataset (see function above)
# Training
nb_epoch: 100
batch_size: 2
detect_text: False # useless
detect_quadrangle: False # useless
phi: 3 # selects corresponding backbone BE CARFUL IT AUGMENTS THE NEEDED RAM
weighted_bifpn: True # bi directionnal featured pyramidal network
freeze_bn: False # freeze batchnorm
freeze_backbone: False
snapshot: imagenet # imagenet or a loading path or None
validation: True # computation validation or not
gpu: '0' # format : device_name:gpu1,gpu2
random_transform: False # apply various online random transformation
workers: 7
multiprocessing: True
max_queue_size: 10
train_evaluation: False # compute mAP for subtrain (can take some time).
focal_gamma: 3
```
We hope that it is self explanatory.

## An example of how to wrap the train method defined above :
```python 
import pprint
import yaml
config = yaml.safe_load() # COMPLETE WITH YOUR CONFIG

LOCAL_ROOT_PATH = f"/home/c3/jupyter_root_dir/data/detection/experiments/{config['architecture']}/{config['experiment_name']}"
assert os.path.exists(LOCAL_ROOT_PATH) == False, 'Experiment name already used!'

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(config)

try:
    train(config)
except Exception as e:
    print(e)
    remote_path = os.path.join(
        REMOTE_EXPERIMENT_PATH,
        os.path.basename(LOCAL_ROOT_PATH)
    )
    c3.Client.uploadLocalClientFiles(localPath=LOCAL_ROOT_PATH, dstUrlOrEncodedPath=remote_path)
```

## Logs
You can find in stdout.txt file all the logs of the experiment, mAP computed at each epoch for each class.
It also save tensorboard logs, and snaphot of the model, keeping the best model according to test mAP evaluation.
