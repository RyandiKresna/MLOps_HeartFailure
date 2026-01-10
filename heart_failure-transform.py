import tensorflow as tf
import tensorflow_transform as tft

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

def transformed_name(key):
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.            
    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale numeric features
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # Convert categorical features to one-hot encoding
    for key in CATEGORICAL_FEATURE_KEYS:
        # Use vocabulary transform which will create a vocabulary for each categorical feature
        indices = tft.compute_and_apply_vocabulary(
            inputs[key], 
            vocab_filename=key,
            # Set num_oov_buckets=1 to handle out-of-vocabulary values
            num_oov_buckets=1,
            # Set top_k to match the known vocabulary size plus OOV
            top_k=VOCAB_SIZES[key] + 1
        )
        # Convert to dense tensor and apply one-hot encoding
        indices_dense = tf.cast(indices, tf.int64)
        # Add one to account for OOV bucket
        depth = VOCAB_SIZES[key] + 1
        one_hot = tf.one_hot(indices_dense, depth=depth)
        # Reshape to remove the unnecessary batch dimension
        one_hot_shaped = tf.reshape(one_hot, [-1, depth])
        outputs[transformed_name(key)] = one_hot_shaped

    # Convert label to float32 for classification
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)

    return outputs