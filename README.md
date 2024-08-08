# JSBach-SampleGenerator
Deep Learning project that uses Generative AI through Python language to produce non-existent musical samples from the german composer, musician, orchestral director, chapel master and teacher of the Baroque period, Johann Sebastian Bach.


## **Notes to keep in mind**

Since the project will use sound samples which will be transformed to spectrograms to train the model and then make an inverse transform from the generated spectrogram sample to actual sound, it is worth to note the possible inputs for generation:

* Conditioned Generation
* Autonomus Generation
* Continuation

We now may define the Generative Sound System below:

* First, we take musical samples from the author Johann Sebastian Bach (it could go from short samples to complete melodies)
* From all the spectrogram flavors, we'll use [Mel Spectrograms](https://ketanhdoshi.github.io/Audio-Mel/) since they perform better for this Deep Learning Model.
* Variational Autoencoders (VAEs) will be used as well as the main architecture of the model.
* From the inputs for generation, we'll be using **Autonomous Generation**.
