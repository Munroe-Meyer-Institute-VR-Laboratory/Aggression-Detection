import enum

import numpy as np
import tensorflow as tf
from keras.layers import Bidirectional
from sklearn.metrics import f1_score, recall_score, precision_score
from tensorflow import keras
from keras import layers, optimizers, losses, applications, regularizers

from hog_testing import HOG_Descriptor


class FeatureExtractorSpec:
    DENSENET = 0  # 1920
    XCEPTION = 1  # 2048
    INCEPTIONRESNETV2 = 2  # 1536
    NASNETLARGE = 3  # 4032
    EFFICIENTNETV2L = 4  # 1280
    CONVNEXTXLARGE = 5  # 2048
    RESNET152V2 = 6  # 2048
    VGG16 = 7  # 512
    NONE = 8
    HOG = 9


FeatureExtractorFeatures = [1920, 2048, 1536, 4032, 1280, 2048, 2048, 512, None, 2304]


def build_feature_extractor(spec, image_size):
    if spec == FeatureExtractorSpec.DENSENET:
        feature_extractor = applications.DenseNet201(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.densenet.preprocess_input
    elif spec == FeatureExtractorSpec.XCEPTION:
        feature_extractor = applications.Xception(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.xception.preprocess_input
    elif spec == FeatureExtractorSpec.VGG16:
        feature_extractor = applications.VGG16(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.vgg16.preprocess_input
    elif spec == FeatureExtractorSpec.NASNETLARGE:
        feature_extractor = applications.NASNetLarge(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.nasnet.preprocess_input
    elif spec == FeatureExtractorSpec.RESNET152V2:
        feature_extractor = applications.ResNet152V2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.resnet_v2.preprocess_input
    elif spec == FeatureExtractorSpec.CONVNEXTXLARGE:
        feature_extractor = applications.ConvNeXtXLarge(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.convnext.preprocess_input
    elif spec == FeatureExtractorSpec.EFFICIENTNETV2L:
        feature_extractor = applications.EfficientNetV2L(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.efficientnet_v2.preprocess_input
    elif spec == FeatureExtractorSpec.INCEPTIONRESNETV2:
        feature_extractor = applications.InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        preprocess_input = keras.applications.inception_resnet_v2.preprocess_input
    elif spec == FeatureExtractorSpec.HOG:
        feature_extractor = HOG_Descriptor(cell_size=64, bin_size=64)
    else:
        raise ValueError("Invalid Feature Extractor spec")
    if spec != FeatureExtractorSpec.HOG:
        inputs = keras.Input((image_size, image_size, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")
    else:
        return feature_extractor


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            embeddings_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
            activity_regularizer=regularizers.L2(1e-3),
            embeddings_constraint=keras.constraints.MaxNorm(3),
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3,
            kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
            bias_regularizer=regularizers.L2(1e-2),
            activity_regularizer=regularizers.L2(1e-3),
            bias_initializer='zeros'
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu,
                          kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
                          bias_regularizer=regularizers.L2(1e-2),
                          activity_regularizer=regularizers.L2(1e-3),
                          kernel_initializer='random_normal',
                          bias_initializer='zeros'
                          ), layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class MulticlassMetrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, writer):
        super(MulticlassMetrics, self).__init__()
        self.validation_data = valid_data
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro', zero_division=0)
        _val_recall = recall_score(val_targ, val_predict, average='macro', zero_division=0)
        _val_precision = precision_score(val_targ, val_predict, average='macro', zero_division=0)

        self.writer.add_scalar('Metrics/F1', _val_f1, epoch)
        self.writer.add_scalar('Metrics/Recall', _val_recall, epoch)
        self.writer.add_scalar('Metrics/Precision', _val_precision, epoch)

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


def compile_cnn_transformer(seq_len, num_features, classes, dense_dim=4, num_heads=10):
    sequence_length = seq_len
    embed_dim = num_features

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = keras.layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes if classes == 1 else classes + 1,
                           activation="sigmoid" if classes == 1 else "softmax",
                           kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
                           bias_regularizer=regularizers.L2(1e-2),
                           activity_regularizer=regularizers.L2(1e-3),
                           kernel_initializer='random_normal',
                           bias_initializer='zeros'
                           )(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy() if classes == 1 else losses.CategoricalCrossentropy(),
        metrics=['accuracy' if classes == 1 else keras.metrics.TopKCategoricalAccuracy(k=1)]
    )
    return model


def compile_time_distributed_rnn_model(seq_len, num_features, batch_size, classes, dense_dim=128, dropout=0.5):
    inputs = tf.keras.Input(shape=(seq_len, num_features))

    x = Bidirectional(keras.layers.GRU(units=dense_dim, activation='tanh', recurrent_activation='sigmoid',
                                       kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
                                       bias_regularizer=regularizers.L2(1e-2),
                                       activity_regularizer=regularizers.L2(1e-3),
                                       stateful=False))(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(classes if classes == 1 else classes + 1,
                                 activation="relu",
                                 kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
                                 bias_regularizer=regularizers.L2(1e-2),
                                 activity_regularizer=regularizers.L2(1e-3),
                                 )(x)

    # x = keras.layers.TimeDistributed(nn)(inputs)
    # x = keras.layers.BatchNormalization()(x)
    # outputs = keras.layers.GRU(units=classes if classes == 1 else classes + 1,
    #                            activation='tanh' if classes == 1 else 'softmax',
    #                            kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
    #                            bias_regularizer=regularizers.L2(1e-2),
    #                            activity_regularizer=regularizers.L2(1e-3),
    #                            )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy() if classes == 1 else losses.CategoricalCrossentropy(),
        metrics=['accuracy' if classes == 1 else keras.metrics.TopKCategoricalAccuracy(k=1)]
    )
    return model


def compile_time_distributed_model(seq_len, num_features, batch_size, classes, dense_dim=128, dropout=0.5):
    inputs = tf.keras.Input(shape=(seq_len, num_features))

    nn = keras.layers.Dense(dense_dim,
                            activation="relu",
                            kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
                            bias_regularizer=regularizers.L2(1e-2),
                            activity_regularizer=regularizers.L2(1e-3),
                            )

    x = keras.layers.TimeDistributed(nn)(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.GRU(units=classes if classes == 1 else classes + 1,
                               activation='tanh' if classes == 1 else 'softmax',
                               kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2),
                               bias_regularizer=regularizers.L2(1e-2),
                               activity_regularizer=regularizers.L2(1e-3),
                               )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy() if classes == 1 else losses.CategoricalCrossentropy(),
        metrics=['accuracy' if classes == 1 else keras.metrics.TopKCategoricalAccuracy(k=1)]
    )
    return model
