import os 
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras.optimizers import Adam, SGD

from generators.csv_ import CSVGenerator
from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet
from losses import smooth_l1, focal, smooth_l1_quad
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

from contextlib import redirect_stdout
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def diff(start, end):
    t_diff = relativedelta(end, start)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def create_callbacks(
    training_model,
    prediction_model,
    evaluation_generator,
    validation_generator,
    logs_path,
    snapshots_path,
    config
):

    callbacks = []

    tensorboard_callback = None
    if logs_path:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir                = logs_path,
            histogram_freq         = 0,
            batch_size             = config['batch_size'],
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)
    
    from eval.pascal import Evaluate
    val_prefix = 'val_'
    evaluation = Evaluate(
        validation_generator,
        prediction_model,
        iou_threshold=0.1,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        weighted_average=False,
        verbose=1,
        tensorboard=tensorboard_callback,
        prefix=val_prefix
    )
    callbacks.append(evaluation)
    if 'train_evaluation' in config and config['train_evaluation']==True:
        evaluation2 = Evaluate(
            evaluation_generator,
            prediction_model,
            iou_threshold=0.1,
            score_threshold=0.05,
            max_detections=100,
            save_path=None,
            weighted_average=False,
            verbose=1,
            tensorboard=tensorboard_callback,
            prefix='train_'
        )
        callbacks.append(evaluation2)
    #h save the model
    # ensure directory created first; otherwise h5py will error after epoch.
    os.makedirs(snapshots_path, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            snapshots_path,
            '{phi}_{{epoch:02d}}.h5'.format(phi=config['phi'])
        ),
        verbose=1,
        save_best_only=True,
        monitor=val_prefix+"mAP",
        mode='max'
    )
    callbacks.append(checkpoint)

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=val_prefix+'mAP', min_delta=0, patience=10, verbose=0,
        mode='max', baseline=None, restore_best_weights=False
    )
    callbacks.append(early_stopping_callback)
            
    return callbacks

def load_efficient_det(
    config,
    LOCAL_ANNOTATIONS_PATH,
    LOCAL_ROOT_PATH,
    LOCAL_CLASSES_PATH,
    LOCAL_VALIDATIONS_PATH,
    LOCAL_LOGS_PATH,
    LOCAL_SNAPSHOTS_PATH
):

    common_args = {
        'phi': config['phi'],
        'detect_text': config['detect_text'],
        'detect_quadrangle': config['detect_quadrangle']
    }

    # create random transform generator for augmenting training data
    if config['random_transform']:
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None

    annotations_df = pd.read_csv(LOCAL_ANNOTATIONS_PATH, header=None)
    # stratified sampling
    N = int(len(annotations_df)*0.15)
    evaluation_df = annotations_df.groupby(5, group_keys=False).apply(
        lambda x: x.sample(int(np.rint(N*len(x)/len(annotations_df))))).sample(frac=1)
    evaluation_path = f'{LOCAL_ROOT_PATH}/evaluation.csv'
    evaluation_df.to_csv(evaluation_path, index=False, header=None)

    config['steps_per_epoch'] = 3#annotations_df.iloc[:,0].nunique()/config['batch_size'] TODO

    train_generator = CSVGenerator(
                LOCAL_ANNOTATIONS_PATH,
                LOCAL_CLASSES_PATH,
                batch_size=config['batch_size'],        
                misc_effect=misc_effect,
                visual_effect=visual_effect,
                **common_args
            )
    evaluation_generator = CSVGenerator(
                evaluation_path,
                LOCAL_CLASSES_PATH,
                batch_size=config['batch_size'],        
                misc_effect=misc_effect,
                visual_effect=visual_effect,
                **common_args
            )
    if config['validation']:
        validation_generator = CSVGenerator(
                    LOCAL_VALIDATIONS_PATH,
                    LOCAL_CLASSES_PATH,
                    batch_size=config['batch_size'],        
                    misc_effect=misc_effect,
                    visual_effect=visual_effect,
                    **common_args
        )
    else:
        validation_generator = None
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

            
    model, prediction_model = efficientdet(
        config['phi'],
        num_classes=num_classes,
        num_anchors=num_anchors,
        weighted_bifpn=config['weighted_bifpn'],
        freeze_bn=config['freeze_bn'],
        detect_quadrangle=config['detect_quadrangle']
    )

    # freeze backbone layers
    if config['freeze_backbone']:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][config['phi']]):
            model.layers[i].trainable = False
    # optionally choose specific GPU
    gpu = config['gpu']
    device = gpu.split(':')[0]
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    if gpu and len(gpu.split(':')) > 1:
        gpus = gpu.split(':')[1]
        model = tf.keras.utils.multi_gpu_model(model, gpus=list(map(int, gpus.split(','))))

    if config['snapshot'] == 'imagenet':
        model_name = 'efficientnet-b{}'.format(config['phi'])
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        model.load_weights(weights_path, by_name=True)
    elif config['snapshot']:
        print('Loading model, this may take a second...')
        model.load_weights(config['snapshot'], by_name=True)

    return (
        model,
        prediction_model,
        train_generator,
        evaluation_generator,
        validation_generator,
        config
    )


def efficientdet_training(
    config,
    LOCAL_ANNOTATIONS_PATH,
    LOCAL_ROOT_PATH,
    LOCAL_CLASSES_PATH,
    LOCAL_VALIDATIONS_PATH,
    LOCAL_LOGS_PATH,
    LOCAL_SNAPSHOTS_PATH
):


    (   model,
        prediction_model,
        train_generator,
        evaluation_generator,
        validation_generator,
        config
    ) = load_efficient_det(
        config,
        LOCAL_ANNOTATIONS_PATH,
        LOCAL_ROOT_PATH,
        LOCAL_CLASSES_PATH,
        LOCAL_VALIDATIONS_PATH,
        LOCAL_LOGS_PATH,
        LOCAL_SNAPSHOTS_PATH
    )
    #model.summary()

    # create the callbacks
    if config['validation']:
        callbacks = create_callbacks(
            model,
            prediction_model,
            evaluation_generator,
            validation_generator,
            LOCAL_LOGS_PATH,
            LOCAL_SNAPSHOTS_PATH,
            config
        )
    else:
        callbacks = create_callbacks(
                model,
                prediction_model,
                evaluation_generator,
                train_generator,
                LOCAL_LOGS_PATH,
                LOCAL_SNAPSHOTS_PATH,
                config
            )
    model.compile(
        optimizer=Adam(lr=1e-3),
        loss={
            'regression': smooth_l1(),
            'classification': focal()
        }
    )

    os.makedirs(LOCAL_LOGS_PATH, exist_ok=True)
    with open(os.path.join(LOCAL_LOGS_PATH, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)
    with open(LOCAL_LOGS_PATH + '/stdout.txt', 'w') as f:
        with redirect_stdout(f):
            print('Path : ', LOCAL_ROOT_PATH)
            start = datetime.now()
            print(f'Started training at: {start}')
            if not validation_generator:
                results = model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=config['steps_per_epoch'],
                    initial_epoch=0,
                    epochs=config['nb_epoch'],
                    verbose=1,
                    callbacks=callbacks,
                    workers=config['workers'],
                    use_multiprocessing=config['multiprocessing'],
                    max_queue_size=config['max_queue_size']
                )
            else:
                results = model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=config['steps_per_epoch'],
                    initial_epoch=0,
                    epochs=config['nb_epoch'],
                    verbose=1,
                    callbacks=callbacks,
                    workers=config['workers'],
                    use_multiprocessing=config['multiprocessing'],
                    max_queue_size=config['max_queue_size'],
                    validation_data=validation_generator
                )
            end = datetime.now()
            print(f'Completed training at: {end}')
            print(f'Total training time: {diff(start, end)}')
