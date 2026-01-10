import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'HeartDisease'

NUMERIC_FEATURE_KEYS = [
    'Age',
    'FastingBS',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]

CATEGORICAL_FEATURE_KEYS = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

# Define vocabulary sizes for categorical features
VOCAB_SIZES = {
    'Sex': 2,  # ['F', 'M']
    'ChestPainType': 4,  # ['ATA', 'NAP', 'ASY', 'TA']
    'RestingECG': 3,  # ['Normal', 'ST', 'LVH']
    'ExerciseAngina': 2,  # ['N', 'Y']
    'ST_Slope': 3,  # ['Up', 'Flat', 'Down']
}

NUM_EPOCHS = 20
BATCH_SIZE = 32

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Load compressed dataset"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


# Focal Loss Implementation for Imbalanced Data
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss untuk menangani class imbalance.
    Formula: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        alpha: Weighting factor untuk positive class (default: 0.25)
        gamma: Focusing parameter untuk mengurangi loss dari easy examples (default: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions untuk menghindari log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Hitung cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Hitung focal loss
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=32) -> tf.data.Dataset:
    """Creates input dataset from transformed data"""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec = {}

    # Add numeric features
    for key in NUMERIC_FEATURE_KEYS:
        feature_spec[transformed_name(key)] = transform_feature_spec[transformed_name(key)]

    # Add categorical features
    for key in CATEGORICAL_FEATURE_KEYS:
        feature_spec[transformed_name(key)] = transform_feature_spec[transformed_name(key)]

    # Add label
    feature_spec[transformed_name(LABEL_KEY)] = transform_feature_spec[transformed_name(LABEL_KEY)]

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        shuffle=True
    )

    def split_label(features):
        label = features.pop(transformed_name(LABEL_KEY))
        return features, label

    return dataset.map(split_label)


def build_mlp_model(hp):
    """
    Builds a deep MLP model with hyperparameters from Keras Tuner.
    Includes multiple hidden layers, batch normalization, and dropout.
    """
    # Create input layers
    inputs = {}
    features = []
    
    # Numeric feature inputs
    for key in NUMERIC_FEATURE_KEYS:
        name = transformed_name(key)
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)
        features.append(inputs[name])
    
    # Categorical feature inputs
    for key in CATEGORICAL_FEATURE_KEYS:
        name = transformed_name(key)
        inputs[name] = tf.keras.Input(shape=(VOCAB_SIZES[key] + 1,), name=name, dtype=tf.float32)
        features.append(inputs[name])
    
    # Concatenate all features
    x = layers.concatenate(features)
    
    # Tunable MLP architecture
    num_layers = hp.Int('num_layers', min_value=2, max_value=4, step=1)
    
    # First layer - larger
    units = hp.Int('units_0', min_value=64, max_value=256, step=64)
    x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hp.Float('dropout_0', 0.2, 0.5, step=0.1))(x)
    
    # Additional hidden layers with decreasing units
    for i in range(1, num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=128, step=32)
        x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1))(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Tunable learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
    
    # Choose loss function
    use_focal_loss = hp.Boolean('use_focal_loss', default=True)
    
    if use_focal_loss:
        # Tunable focal loss parameters
        alpha = hp.Float('focal_alpha', 0.25, 0.75, step=0.25)
        gamma = hp.Float('focal_gamma', 1.0, 3.0, step=0.5)
        loss = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        loss = 'binary_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Defines the Keras Tuner search with MLP and Focal Loss."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create datasets
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=15, batch_size=BATCH_SIZE)
    eval_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=15, batch_size=BATCH_SIZE)

    # Use RandomSearch with reasonable trials
    tuner = kt.RandomSearch(
        hypermodel=build_mlp_model,
        objective=kt.Objective('val_auc', direction='max'),  # AUC lebih baik untuk imbalanced data
        max_trials=30,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='heart_failure_mlp_focal_tuning',
        overwrite=True,
        max_retries_per_trial=1
    )

    # Callbacks untuk training yang lebih stabil
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 8,
            "callbacks": callbacks,
            "verbose": 1
        }
    )