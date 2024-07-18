# housenumbercnn

# Setting up the data
In this project, I built a convolutional neural network to detect and classify house address numbers. I used Pytorch to do this, and was able to get an accurate prediction ~91% of the time. First, I downloaded the data set and split the images into training, testing, and validation data, transforming the data into tensors and also normalizing the images to increase accuracy. I also created data loaders for my model so I could process the data in batch sizes when I ran my neural network.

# My model
The first convolutional layer of my model took in 3 input channels (red, green, blue) and outputted 32 output channels (feature maps), with a 3x3 filter/kernel and a padding of 1. I made my model with three convolution layers, with each layer taking in the previous number of input channels and doubling it, and in between each layer, I pooled the layer to downsample and also applied the ReLU activation function to assess a neuron's impact. After three convolution layers, I flattened the tensor and implemented the feedforward part of my model, which used dropout and ReLU to reduce the model to 10 neurons for each of the digits. 

To train the model, I fiddled around with different optimizers and loss functions, but ultimately decided to use the Adam optimizer and cross entropy loss function for the best results. For the validation step, I tested my trained model on the validation data for five epochs before ultimately evaluating my model on the testing data. Here are the results from an example run of my model:

Using downloaded and verified file: data/train_32x32.mat
Using mps device
Epoch 1/5, Training Loss: 1.1035491724390734, Validation Loss: 0.48674971823181423, Accuracy: 85.13%
Epoch 2/5, Training Loss: 0.4650842060070289, Validation Loss: 0.3943548527147089, Accuracy: 88.14%
Epoch 3/5, Training Loss: 0.3780051958796225, Validation Loss: 0.3702607806391862, Accuracy: 89.13%
Epoch 4/5, Training Loss: 0.33124305274925736, Validation Loss: 0.35430812964938124, Accuracy: 89.60%
Epoch 5/5, Training Loss: 0.29555854998921094, Validation Loss: 0.3306947190846716, Accuracy: 91.66%
Using downloaded and verified file: data/test_32x32.mat
Test Error: 
 Accuracy: 90.9%, Avg loss: 0.332268 
