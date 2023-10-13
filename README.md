# IMAGE_COLORISATION


Given a grayscale photograph as input, the problem of hallucinating a plausible color version of the photograph. This problem is clearly underconstrained, so previous approaches have either relied on significant user interaction or resulted in desaturated colorizations. We propose a fully automatic approach that produces vibrant and realistic colorizations. We embrace the underlying uncertainty of the problem by posing it as a classification task and use class-rebalancing at training time to increase the diversity of colors in the result. The system is implemented as a feed-forward pass in a CNN at test time and is trained on over a million color images.


We design and build a convolutional neural network (CNN) that accepts a black-and-white image as an input and generates a colorized version of the image as its output;We’re going to use the Caffe colourization model for this program. And you should be familiar with basic OpenCV functions and uses like reading an image or how to load a pre-trained model using dnn module etc.Basically like RGB colour space, there is something similar, known as Lab colour space. And this is the basis on which our program is based.RGB, lab colour has 3 channels L, a, and b. But here instead of pixel values, these have different significances i.e : 
L-channel: light intensity                                  
a-channel: green-red encoding
b- channel: blue-red encoding
