import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, applications
from experiment_setup import DenseNetSpec


# Following method is modified from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_video(path, image_size, max_frames, vocab, output):
    cap = cv2.VideoCapture(path)
    frames = []
    raw_frames = []
    predictions = []
    counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
            frame = cv2.resize(frame, (image_size, image_size))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                frames_seq = prepare_single_video(np.array(frames), max_frames, 1920)
                event_time = ((float(len(predictions)) * 16.0) + 8.0) / float(fps)
                event_pred = predict_action(model, frames_seq)
                predictions.append(event_pred)
                if event_pred[0] > 0.7:
                    frame_width = int(cap.get(3))
                    frame_height = int(cap.get(4))
                    out_file = f"{pathlib.Path(path).stem}_{event_pred[0]:.4f}_{counter}.mp4"
                    dp_file = f"{pathlib.Path(path).stem}.txt"
                    out = cv2.VideoWriter(os.path.join(output, out_file),
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                          (frame_width, frame_height))
                    for frame in raw_frames:
                        out.write(frame)
                    out.release()
                    counter += 1
                    print(f"\nEvent Count: {counter}")
                    for i in np.argsort(event_pred):
                        print(f"  {vocab[i]}: {event_pred[i] * 100:5.2f}%")
                    with open(os.path.join(output, dp_file), 'a') as f:
                        f.write(f'"Freq","a","hitting",{event_time:.1f}\n')
                frames = []
                raw_frames = []
    finally:
        cap.release()


def load_video_windowed(path, image_size, max_frames, vocab, output, window_size):
    cap = cv2.VideoCapture(path)
    frames = []
    raw_frames = []
    predictions = []
    windows = []
    counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
            frame = cv2.resize(frame, (image_size, image_size))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                frames_seq = prepare_single_video(np.array(frames), max_frames, 1920)
                event_time = ((float(len(predictions)) * 16.0) + 8.0) / float(fps)
                event_pred = predict_action(model, frames_seq)
                predictions.append(event_pred)
                windows.append(event_pred)
                if len(windows) == window_size:
                    window_pred = 0.
                    for window in windows:
                        window_pred += window[0]
                    window_pred /= float(window_size)
                    print(f"Windows: {windows} | Window Pred: {window_pred}")
                    if window_pred > 0.7:
                        frame_width = int(cap.get(3))
                        frame_height = int(cap.get(4))
                        out_file = f"{pathlib.Path(path).stem}_{event_pred[0]:.4f}_{counter}.mp4"
                        dp_file = f"{pathlib.Path(path).stem}.txt"
                        out = cv2.VideoWriter(os.path.join(output, out_file),
                                              cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                              (frame_width, frame_height))
                        for frame in raw_frames:
                            out.write(frame)
                        out.release()
                        counter += 1
                        print(f"\nEvent Count: {counter}")
                        for i in np.argsort(event_pred):
                            print(f"  {vocab[i]}: {event_pred[i] * 100:5.2f}%")
                        with open(os.path.join(output, dp_file), 'a') as f:
                            f.write(f'"Freq","a","hitting",{event_time:.1f}\n')
                    windows = []
                frames = []
                raw_frames = []
    finally:
        cap.release()


def prepare_single_video(frames, max_seq, num_features):
    # For each video.
    # Gather all its frames and add a batch dimension.

    frames = frames[None, ...]

    # Initialize placeholder to store the features of the current video.
    frame_features = np.zeros(
        shape=(1, max_seq, num_features), dtype="float32"
    )

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(max_seq, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )

            else:
                frame_features[i, j, :] = 0.0

    return frame_features


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
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
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim), ]
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


def get_compiled_model(seq_len, num_features):
    sequence_length = seq_len
    embed_dim = num_features
    dense_dim = 64
    num_heads = 16
    lstm_units = 16
    classes = 3

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adamax(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    return model


def get_compiled_model_2(seq_len, num_features):
    sequence_length = seq_len
    embed_dim = num_features
    dense_dim = 64
    num_heads = 16
    lstm_units = 16
    classes = 1

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy(),
        metrics=["accuracy"])
    return model


def build_feature_extractor(spec, image_size):
    if spec == DenseNetSpec.DENSENET121:
        feature_extractor = applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
    elif spec == DenseNetSpec.DENSENET169:
        feature_extractor = applications.DenseNet169(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
    elif spec == DenseNetSpec.DENSENET201:
        feature_extractor = applications.DenseNet201(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
    else:
        raise ValueError("Invalid DenseNet spec")
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((image_size, image_size, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def predict_action(model, frame_features):
    probabilities = model.predict(frame_features).reshape(-1).tolist()
    return probabilities


if __name__ == '__main__':
    video_paths = [

    ]
    vocab = ['hitting', 'nobehavior', 'throwingobj']
    # Get latest weights and load into model
    checkpoint_dir = r''
    weights_dir = pathlib.Path(checkpoint_dir)
    weights_pattern = r'*.hdf5'
    latest_weight = max(weights_dir.glob(weights_pattern), key=lambda f: f.stat().st_ctime)
    print(f"Loaded weights for testing: {latest_weight}")
    # Test model using test data
    densenet_spec = DenseNetSpec.DENSENET201
    feature_extractor = build_feature_extractor(densenet_spec, 1024)
    model = get_compiled_model_2(16, 1920)
    model.load_weights(os.path.join(checkpoint_dir, latest_weight))
    for video_path in video_paths:
        output_path = os.path.join(r"experiment_results\Inference", pathlib.Path(video_path).stem)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        load_video_windowed(video_path, 1024, 16, vocab, output_path, 2)
