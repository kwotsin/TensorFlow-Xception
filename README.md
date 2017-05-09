# tensorflow-xception
TensorFlow implementation of the Xception Model by François Chollet, based on the paper:
[Xception: Deep Learning with Depthwise Separable Convolutions.](https://arxiv.org/abs/1610.02357)

As an example, the model will be trained on the Flowers dataset.

### Contents

1. **xception.py**: The model implementation file.
2. **xception-preprocessing.py**: This is the exact same preprocessing used for inception models.
3. **xception_test.py**: A test file to check for the correctness of the model implementation. Can be executed by itself.
4. **write_pb.py**: A file to freeze your graph for inference purposes after training your model.
5. **train_flowers.py**: An example script to train an Xception model on the flowers dataset.
6. **dataset**: A folder containing the flowers dataset prepared in TFRecords format.

### How to run

1. run `python train_flowers.py` from the root directory to start training your Xception model from scratch on the Flowers dataset. A log directory will be created.
2. run `tensorboard --logdir=log` on the root directory to get your tensorboard visualizations.
3. Tweak around with the hyperparameters and have fun! :D

### Customization

You can simply change the dataset files and the appropriate names (i.e. anything that has the name 'flowers') to use the network for your own purposes. Importantly, you should be able to obtain the TFRecord files for your own dataset to start training as the data pipeline is dependent on TFRecord files. To learn more about preparing a dataset with TFRecord files, see this [guide](https://github.com/kwotsin/create_tfrecords) for a reference.


### References

1. [Xception: Deep Learning with Depthwise Separable Convolutions.](https://arxiv.org/abs/1610.02357)
2. [Code from the transfer learning guide](https://github.com/kwotsin/transfer_learning_tutorial)
3. [Keras implementation of the model](https://github.com/fchollet/deep-learning-models/blob/master/xception.py)
4. [TF-Slim layers reference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py)
