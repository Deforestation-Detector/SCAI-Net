# A simplified description of the model architecture:
* We used a neural network to make predictions on images of the Amazon rainforest. More specifically, we used a convolutional ensemble neural network. 
  * A convolutional neural network (CNN) is fantastic at finding distinguishing characteristics of two or more things. In our case, a CNN tells the difference between satellite images of the Amazon rainforest by looking at the entire image, pixel by pixel. A CNN gets its name from the math used to find the distinguishing characteristics which help the network tell two or more images apart: convolutions. 
  * An ensemble network is a collection of multiple networks where all the networks's predictions are combined in some way. Our ensemble network combined the predictions of multiple CNNs and averaged the predictions of each one. 

# An in-depth description of the model architecture:
