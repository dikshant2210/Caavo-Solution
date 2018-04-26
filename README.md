# Caavo-Solution

File Structure for the dataset:
    input/
        dataset/
            train/
            test/
            train_features/

train/ -> contains all the train images
test/ -> contains all the test images
train_features/ -> contain convolutional activations for the training set

** Out of the 62258 training images 52258 were used for training and 10000 for
   validation.

Explanation of the solution:
    My approach is based on transfer learning. Since large deep neural networks
    cannot be trained on small amounts of data, I used pretrained models on the
    imagenet dataset. Particularly i used the convolutional features of the
    VGG19 model.
    My is a two step solution:
    Step 1:
        Removed all the fully connected layers from the VGG19 model and replace
        them with new fully connected layers of different number of neurons
        from the original VGG19 model. For quick prototyping of this step i
        precalculated the VGG19 convolutional activations and used them to train
        a fully connected deep neural network. Throuout this step no convloutional
        layer is trained. Convloutional activations are passed to the fully
        fully connected model using GlobalAveragePooling.
        Observations:
            I tried different optimizers and network architecture to train the
            model. Best results were found using 4096->2048->15 architecture
            trained using Stochastic Gradient Descent with dropout in between
            the hidden layers. Learning rate was reduced by a factor of 0.1
            if there was no improvement in three successive epochs.
            Using sigmoid activations for the dense layer activations performed
            better than relu activations.

    Step 2:
        Above step gave me a leaderboard score of about 0.62. To further improve
        the model I made the last convolutional layer of VGG19 trainable as well.
        With a very low learning rate of 0.0001 model was trained using
        Stochastic Gradient Descent with momentum of 0.9. As in the above step,
        learning rate was reduced by a factor of 0.1 if there was no improvement
        in three successive epochs. Finetuning the last convloutional layer gave
        me a leaderboard score of 0.646 .

Source Files:
    All the source files are ipython notebooks. Python files are just Python
    downloads of the notebooks.
    **quick.ipynb/quick.py -> Source code for training fully connected model.
    **extract_features.ipynb/ extract_features.py -> Source code for getting and
          storing convloutional activations of the training images.
    **caavo_conv_ft2.ipynb/ caavo_conv_ft2.py -> Source code for finetuning the last
          convolutional layer of VGG19 model.

Weights files:
    **best_weights_4096_0.2_2048_0.5_sigmoid.hdf5 ->
          Weights with only the fully connected top layers trainable.
    **best_weights_4096_0.2_2048_0.5_ft_sigmoid.hdf5 ->
          Weights after finetuning the last convolutional layer.


Model Architecture:
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
global_average_pooling2d_1 ( (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_2 (Dense)              (None, 2048)              8390656
_________________________________________________________________
dropout_2 (Dropout)          (None, 2048)              0
_________________________________________________________________
dense_3 (Dense)              (None, 15)                30735
=================================================================
