# Art Deep Learning

This git contains a serie of Neural Network dedicated to arts task, such as style classification and image generation.

### Link to datasets used in the notebooks:

- https://drive.google.com/file/d/1XJUX7dvY63WkouPPOtXObqPWwzbtC-k0/view?usp=sharing
- https://drive.google.com/file/d/1YEjLO21Ue3TbUl9rbPlW8Ebd-mBMDWza/view?usp=sharing
- https://drive.google.com/file/d/1esXAbZgLWKJ2BB0By98fhrgPsOdHRlLG/view?usp=sharing

## Style classification

![Style-classification](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/impressionism-classification.jpg)

Style classification notebook present a model to detect impressionisms artworks. The idea is to use a pre-trained model (in this case, ResNet and Inception got the best performance) and only do a fine-tuning.

## Image Auotencoder

![Autoencoder-example](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Autoencoder-example.jpg)

Auotencoder-artwork notebook present a model to auotencode the artworks. The objective is to get a low dimensional representation that catch the main features from each one. There are two models:
- Denoising Autoencoder
- Sliced- Wasserstein Autoencoder
