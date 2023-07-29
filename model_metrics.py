import time
import timeit

import keras.backend
import tensorflow as tf
import keras.backend as K
from cnn_transformer import compile_cnn_transformer, FeatureExtractorSpec, scheduler, \
    compile_time_distributed_model, compile_time_distributed_rnn_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph, \
    convert_variables_to_constants_v2


# def get_flops(model):
#     concrete = tf.function(lambda inputs: model(inputs))
#     concrete_func = concrete.get_concrete_function(
#         [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
#     frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
#     with tf.Graph().as_default() as graph:
#         tf.graph_util.import_graph_def(graph_def, name='')
#         run_meta = tf.compat.v1.RunMetadata()
#         opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#         flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
#         return flops.total_float_ops

def get_flops(model, model_inputs) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """
    # if not hasattr(model, "model"):
    #     raise wandb.Error("self.model must be set before using this method.")

    if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
            .with_empty_output()
            .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2


def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def measure_timing(model):
    dummy_input = keras.backend.random_normal((128, 16, 2048))
    repetitions = 100
    total_time = 0
    for rep in range(repetitions):
        start_time = time.time()
        model(dummy_input)
        total_time += (time.time() - start_time) / 128
    return total_time / repetitions




weights_file = r"E:\Thesis Results\Keypoint-LSTM\MLHC\All\Tracklet_0_5_1_hitting_None\checkpoints\2_weights.187-0.9163.hdf5"
model = compile_cnn_transformer(16, 2048, 1)
# weights_file = r"E:\Thesis Results\Keypoint-LSTM\MLHC\All\Tracklet_3_5_1_hitting_None\checkpoints\2_weights.100-0.9059.hdf5"
# model = compile_time_distributed_model(16, 2048, 128, 1)
# model.load_weights(weights_file)

print(f"\n\nExecution Time: {measure_timing(model)*1000} ms")
print(f"FLOPS: {get_flops(model, keras.backend.random_normal((1, 1, 16, 2048)))}")
print(f"Memory: {get_model_memory_usage(128, model)*1000} MB")
print(f"Params: {model.count_params()}")
