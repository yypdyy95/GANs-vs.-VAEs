Demos
===
There are four different demonstrations of GANs and VAEs here. 

# 1. mnist_demo.py
This one demonstrates the 2D latent space of an VAE for the MNIST dataset. On the left site this latent space is shown, while on the right site the corresponding number is displayed. Points printed in latent space correspond to the real data samples. 
![mnist_demo](https://user-images.githubusercontent.com/26285498/28509712-47bde4e6-7044-11e7-89d7-d41b763ae2b4.gif)

# 2. mnist_con_demo.py
Again you see the 2D latent space of an VAE for MNIST. Except this one is a conditional Autoencoder, so by scolling you can change the number which will be displayed. 
![mnist_con_demo](https://user-images.githubusercontent.com/26285498/28509713-49ac8abe-7044-11e7-8394-cc8a6818d6a0.gif)

# 3. char_demo.py
Here you see the latent space for a Auxillary-Wasserstein-GAN. This one works with the char74k dataset. In this case the latent space is 100 dimensional and is displayed by a 10 by 10 image with different greyscales, where you can paint in (use: scroll to change color, press left mouse key to assign the pixel a new color). Don't be confused if nearly nothing changes in the output while painting. Only very few pixels will have influence on the output for different characters. 

![char_demo](https://user-images.githubusercontent.com/26285498/28509716-4b76d71e-7044-11e7-8e0e-726f0544e6a3.gif)

# 4. celeb_demo.py
Here a conditional VAE for the celebA dataset is demonstrated. Two conditions can be changed, which are the male and smiling feature from the annotations of the dataset. By checking the according box in the bottom left you can manipulate these features in the images. 


![celeb_demo](https://user-images.githubusercontent.com/26285498/28509718-4d219982-7044-11e7-8fb2-43e7cabe5fdb.gif)
