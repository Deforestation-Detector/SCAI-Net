# A simplified description of the model architecture:
* we used a number of transfer learning models consisting of convolutional neural networks, and created an ensemble of said networks to obtain predictions on images of the Amazon rainforest.
  * Transfer learning consists of starting with a model that has been known to perform well, and reconfiguring it to interact with new data.
  * A CNN is fantastic at recognizing patterns. For example, in our case, the CNN might learn that squiggly and curved shapes indicate roads or water are present in the image. A CNN gets its name from the math used to find the distinguishing patterns which help the network tell two or more images apart: convolutions.
  * An ensemble network is a collection of multiple networks where all the networks' predictions are combined in some way. Our ensemble network combined the predictions of multiple CNNs and averaged the predictions of each one.
* Should you wish to learn more about our architecture check out the Architecture.tex file in our repository:

# An in-depth description of the model architecture:
