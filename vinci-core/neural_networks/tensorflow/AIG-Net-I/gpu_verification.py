import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
