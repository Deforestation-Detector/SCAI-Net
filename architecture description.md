# A simplified description of the model architecture:
* We used a neural network to make predictions on images of the Amazon rainforest. More specifically, we used a transfer learning, convolutional, ensemble neural network. 
  * A convolutional neural network (CNN) is fantastic at recognizing patterns. For example, in our case, the CNN might learn that squiggly and curved shapes indicate roads or water are present in the image. A CNN gets its name from the math used to find the distinguishing patterns which help the network tell two or more images apart: convolutions. 
  * Transfer learning consists of . Transfer learning consists of starting with a model that has been known to perform well, and reconfiguring it to interact with new data.
  * An ensemble network is a collection of multiple networks where all the networks's predictions are combined in some way. Our ensemble network combined the predictions of multiple CNNs and averaged the predictions of each one.

# An in-depth description of the model architecture:
