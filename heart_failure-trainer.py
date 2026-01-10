import json, os
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_transform as tft
from typing import Dict, List, Text
from tfx.components.trainer.fn_args_utils import FnArgs
import warnings
warnings.filterwarnings("ignore")

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

LABEL_KEY = 'HeartDisease'

# Training constants
NUM_EPOCHS = 50
BATCH_SIZE = 32

# Define vocabulary sizes for categorical features
VOCAB_SIZES = {
    'Sex': 2,  # ['F', 'M']
    'ChestPainType': 4,  # ['ATA', 'NAP', 'ASY', 'TA']
    'RestingECG': 3,  # ['Normal', 'ST', 'LVH']
    'ExerciseAngina': 2,  # ['N', 'Y']
    'ST_Slope': 3,  # ['Up', 'Flat', 'Down']
}

def transformed_name(key):
    """Renaming transformed features"""
    return key + '_xf'


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


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=BATCH_SIZE) -> tf.data.Dataset:
    """Creates input dataset from transformed data"""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Create feature spec for transformed features
    feature_spec = {}
    
    # Add numeric features
    for key in NUMERIC_FEATURE_KEYS:
        feature_name = transformed_name(key)
        feature_spec[feature_name] = transform_feature_spec[feature_name]
    
    # Add categorical features
    for key in CATEGORICAL_FEATURE_KEYS:
        feature_name = transformed_name(key)
        feature_spec[feature_name] = transform_feature_spec[feature_name]
    
    # Add label
    feature_spec[transformed_name(LABEL_KEY)] = transform_feature_spec[transformed_name(LABEL_KEY)]
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs
    )
    
    # Separate features and label
    def split_label(features):
        label = features.pop(transformed_name(LABEL_KEY))
        return features, label
        
    # Transform dataset to have (features, label) format
    dataset = dataset.map(split_label)
    
    # Optimize pipeline
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_mlp_model(hp):
    """
    Creates a deep MLP model for heart disease prediction with Focal Loss support.
    
    Args:
        hp: Dict containing best hyperparameters from tuning.
    Returns:
        A compiled Keras Model.
    """
    # Extract hyperparameters
    if isinstance(hp, dict) and 'values' in hp:
        hp = hp['values']
    hp = hp or {}
    
    # Get hyperparameters with defaults
    num_layers = hp.get('num_layers', 3)
    use_focal_loss = hp.get('use_focal_loss', True)
    focal_alpha = hp.get('focal_alpha', 0.25)
    focal_gamma = hp.get('focal_gamma', 2.0)
    learning_rate = hp.get('learning_rate', 0.001)
    
    # Create inputs
    inputs = {}
    feature_tensors = []

    # Handle numeric features
    for key in NUMERIC_FEATURE_KEYS:
        feature_name = transformed_name(key)
        inp = layers.Input(shape=(1,), name=feature_name, dtype=tf.float32)
        inputs[feature_name] = inp
        feature_tensors.append(inp)

    # Handle categorical features
    for key in CATEGORICAL_FEATURE_KEYS:
        feature_name = transformed_name(key)
        inp = layers.Input(shape=(VOCAB_SIZES[key] + 1,), name=feature_name, dtype=tf.float32)
        inputs[feature_name] = inp
        feature_tensors.append(inp)
    
    # Concatenate all features
    x = layers.concatenate(feature_tensors)
    
    # Build MLP layers based on hyperparameters
    for i in range(num_layers):
        units = hp.get(f'units_{i}', 64 if i == 0 else 32)
        dropout_rate = hp.get(f'dropout_{i}', 0.3)
        
        x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Choose loss function
    if use_focal_loss:
        loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"✓ Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
    else:
        loss = 'binary_crossentropy'
        print("✓ Using Binary Crossentropy Loss")
    
    # Compile model with comprehensive metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR')  # Precision-Recall AUC
        ]
    )
    
    model.summary()
    return model


class TransformFeaturesLayer(tf.keras.layers.Layer):
    """A layer for applying TF Transform transformations."""
    def __init__(self, tf_transform_output, **kwargs):
        super().__init__(**kwargs)
        self.tft_layer = tf_transform_output.transform_features_layer()
        
    def call(self, inputs):
        return self.tft_layer(inputs)


class ServingModel(tf.keras.Model):
    """A model wrapper that handles TF Transform preprocessing."""
    def __init__(self, model, tf_transform_output):
        super().__init__()
        self.model = model
        self.transform_layer = TransformFeaturesLayer(tf_transform_output)
    
    def call(self, inputs):
        transformed_features = self.transform_layer(inputs)
        return self.model(transformed_features)


def get_serve_tf_examples_fn(model: tf.keras.Model,
                           tf_transform_output: tft.TFTransformOutput):
    """Returns a function that parses a serialized tf.Example and applies TFT."""
    
    # Create feature spec for raw features
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(LABEL_KEY)
    
    # Create serving model that includes transformation
    serving_model = ServingModel(model, tf_transform_output)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        # Parse raw features
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        return serving_model(parsed_features)
    
    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Train the model and save both serving & eval versions."""
    
    print("="*80)
    print("🚀 Starting Heart Disease Model Training with MLP + Focal Loss")
    print("="*80)
    
    # --- Setup TFT and hyperparameters ---
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    best_hyperparameters = fn_args.hyperparameters
    
    if isinstance(best_hyperparameters, str):
        best_hyperparameters = json.loads(best_hyperparameters)
    
    best_hp = best_hyperparameters.get('values', {}) if best_hyperparameters else {}
    
    print("\n📊 Hyperparameters:")
    for key, value in best_hp.items():
        print(f"  - {key}: {value}")

    # --- Create datasets ---
    print("\n📁 Loading datasets...")
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=20)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)

    # --- Setup model directories ---
    model_dir = fn_args.serving_model_dir
    log_dir = os.path.join(os.path.dirname(model_dir), 'logs')
    serving_model_dir = fn_args.serving_model_dir
    eval_model_dir = os.path.join(os.path.dirname(serving_model_dir), 'Format-Eval')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_model_dir, exist_ok=True)

    # --- Setup callbacks ---
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, 
            update_freq='batch',
            histogram_freq=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(log_dir, 'best_model.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # --- Build and train model ---
    print("\n🏗️  Building MLP model...")
    model = create_mlp_model(best_hp)
    
    print("\n🎯 Training model...")
    history = model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=eval_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # Print final metrics
    print("\n" + "="*80)
    print("📈 Final Training Results:")
    print("="*80)
    final_metrics = history.history
    if final_metrics:
        last_epoch = len(final_metrics.get('loss', []))
        print(f"Training completed in {last_epoch} epochs")
        
        if 'val_accuracy' in final_metrics:
            print(f"  - Val Accuracy: {final_metrics['val_accuracy'][-1]:.4f}")
        if 'val_auc' in final_metrics:
            print(f"  - Val AUC: {final_metrics['val_auc'][-1]:.4f}")
        if 'val_precision' in final_metrics:
            print(f"  - Val Precision: {final_metrics['val_precision'][-1]:.4f}")
        if 'val_recall' in final_metrics:
            print(f"  - Val Recall: {final_metrics['val_recall'][-1]:.4f}")

    # === SAVE SERVING MODEL ===
    print("\n" + "="*80)
    print("💾 Saving Models...")
    print("="*80)
    print("\n🔹 Saving serving model...")
    
    serving_model = ServingModel(model, tf_transform_output)

    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(LABEL_KEY, None)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_fn(serialized_examples):
        parsed_features = tf.io.parse_example(serialized_examples, feature_spec)
        return serving_model(parsed_features)

    print(f"Saving serving model to: {serving_model_dir}")
    tf.saved_model.save(
        serving_model,
        serving_model_dir,
        signatures={'serving_default': serve_fn}
    )
    print("✅ Serving model saved successfully!")

    # === SAVE EVAL MODEL ===
    print("\n🔹 Saving eval model...")
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def eval_fn(serialized_examples):
        parsed_features = tf.io.parse_example(serialized_examples, feature_spec)
        preds = serving_model(parsed_features)
        return {'predictions': preds}

    print(f"Saving eval model to: {eval_model_dir}")
    tf.saved_model.save(
        serving_model,
        eval_model_dir,
        signatures={'serving_default': eval_fn}
    )
    print("✅ Eval model saved successfully!")

    print("\n" + "="*80)
    print("🎉 Model training and saving completed successfully!")
    print("="*80)