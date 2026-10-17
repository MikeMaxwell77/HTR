"""Fail unless CUDA executes work, then exercise the project's training pipeline."""
import sys
import unittest

import tensorflow as tf


def main():
    # The save/load test compares float32 outputs at 1e-6. Disable TensorFloat-32
    # approximations for this precision check; training keeps its normal defaults.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    print(f"TensorFlow: {tf.__version__}", flush=True)
    print(f"Build: {tf.sysconfig.get_build_info()}", flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No TensorFlow GPU detected. Check the driver and Docker GPU passthrough.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(tf.config.experimental.get_device_details(gpu), flush=True)
    # No silent CPU fallback for this operation; numpy() synchronizes execution.
    tf.config.set_soft_device_placement(False)
    with tf.device("/GPU:0"):
        result = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    if "GPU" not in result.device or float(result.numpy()[0, 0]) != 32.0:
        raise RuntimeError("GPU matrix multiplication failed")
    print(f"Matrix multiplication passed on {result.device}", flush=True)
    # CTC has CPU operations, so restore normal placement for the full model.
    tf.config.set_soft_device_placement(True)
    suite = unittest.defaultTestLoader.discover("tests")
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if outcome.wasSuccessful() and not outcome.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
