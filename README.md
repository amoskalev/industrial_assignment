# Clocks vs. Crocodiles
***Test assignment***

In this assignment we will try to solve 3 task:
1. Classify wheather clock or crocodile is on the image
2. Find images that look like both clock and crocodile
3. Generate images that look like both clock and crocodile
># Data

 - 500 images of clocks
 - 500 images of crocodiles
 
[img]

Images are 32x32x3 RGB
># Augmentation

As dataset is tiny I did augmentation based on flips and random_crops.

# Task 1. Classify

I bet we can solve this task with very simple NN, however for task 2 we will need more or less considerable CNN to compute convolutional feature because I want to use the perceptual loss as the measure of closeness of images. So, I will firstly just demonstrate the work of simple NN and the train more complex CNN.

 - Shallow network:

If consider the network that consist only from 4 layers with LeakyReLU's as non-linearities, that network is able to achieve 65 - 75% accuracy. The main advantage of such network is that it's extremly cheap to train.

****For details check tutorial.ipynb notebook***

 - CNN:

With standart CNN, which consist of the sequence of convolutions, LeakyReLU's and maxpoolings we are able to achieve up to 92% accuracy.

In practice, I implemented a network with 5 conv layers and global pooling on the top. However, if we try with only 3 convolutional layers, the result is almost the same. Indeed, I use more convolutions than needed to produce convolutional features that are abstract enough to use in the second task.

    P.S. For both networks I used validation dataset in order to adjust the number of epoches to train.

****For details check tutorial.ipynb notebook***

# Task 2. Finding images that look like both clock and crocodile

For this task I purpose to use so-called perceptual loss. We will take features that some layer of our network produces and define them as the measure of similiarity between images.

Next, if we imagine two clusters of clocks and crocodiles, then we can estimate the centers of these clusters as just the mean of all images (their convolutional features). Then, we can take embeddings of all our images and substitute centers of the cluster. It actually won't tell us what images are both like clock and crocodile, but it will help to rank them in a way like what clocks look more like a crocodile and what crocodiles look more like a clock. Then, in this ranked array of images, we can take images located near the center of and it will return us images that lay clothes to both clusters.

Empiricaly, I found that taking features of the last fully-connected layer (so-called fc7 features), rank images as described above and take them from one of the edges earns best results.

[img]

As we see, several images that this approach finds, are really close to both clock and crocodiles.

One may see that it relies on textures and colors. For example, NN may select images $6,11$ here, because a blue background is much more common for crocodiles because of the water behind. An image with index $8$ may have been selected because the same reason it's too contrast for crocodile and such contrast is common for clocks on the white wall. 

Based on this and on other selected images, one can conclude that this method relies on textures and colors, it's not perfect, but if we can produce more abstract convolutional features with deeper network, I believe it will work good.

        P.S. I also tried an approach with convolutional autoencoder. And again, I was striving to create deep embeddings for images. However, it seems that such autoencoder should be very deep to produce more or less usefull deep features.

****For details check tutorial.ipynb notebook***

# Task 3. Generate half clock, half crocodile
In progress. I try to use conditional dcGAN and VAE. 

dcGAN seems to fail because either I train it in no so appropriate way or such tiny dataset is not enough for dcGAN to converge. An idea is to condition generator and discriminator with class labels of $(0,1)$ and then, when we want to obtain a mix of clock and crocodile, interpolate and condition generator as $(0.5, 0.5)$

I haven't tried VAE yet, but it might actually work. 

    To be continued...
# References:
Parts of code from github.com/yandexdataschool/Practical_DL were used.

