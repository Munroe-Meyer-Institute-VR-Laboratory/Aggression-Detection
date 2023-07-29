import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, losses, regularizers

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# TUBELET EMBEDDING
PATCH_SIZE = (16, 16, 2)

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches


class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim,
            # embeddings_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
            # activity_regularizer=regularizers.L2(1e-3),
            # embeddings_constraint=keras.constraints.MaxNorm(3),
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape,
    num_classes,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1,
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu,
                             ),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu,
                             ),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="sigmoid" if num_classes == 1 else "softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def compile_vivit_model(input_shape, num_classes):
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
        input_shape=input_shape,
        num_classes=num_classes if num_classes == 1 else num_classes + 1
    )

    # Compile the model with the optimizer, loss function and the metrics.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.BinaryCrossentropy() if num_classes == 1 else losses.CategoricalCrossentropy(),
        metrics=['accuracy' if num_classes == 1 else keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    return model
