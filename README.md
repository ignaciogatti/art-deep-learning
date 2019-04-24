# Art Deep Learning

This git contains a serie of Neural Network dedicated to arts task, such as style classification and image generation.

## Dataset

The data for the algorithm was recolected using a crawler.

### Crawler

The crawler take as input a url for a Wikiart query result (in this case a style result: Impressionism, Barroc, etc). Then download all the image result and the metadata associated to them. 
The crawler was developed using Selenium and Scrapy. Selenium was used to deal with the page's dinamic content and Scrapy to crawl only the meaningful content.

![Crawler-description](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/crawler-description.jpg)

### Link to datasets used in the notebooks:

- https://drive.google.com/file/d/1XJUX7dvY63WkouPPOtXObqPWwzbtC-k0/view?usp=sharing
- https://drive.google.com/file/d/1YEjLO21Ue3TbUl9rbPlW8Ebd-mBMDWza/view?usp=sharing
- https://drive.google.com/file/d/1esXAbZgLWKJ2BB0By98fhrgPsOdHRlLG/view?usp=sharing

## Style classification

![Style-classification](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/impressionism-classification.jpg)

Style classification notebook present a model to detect impressionisms artworks. The idea is to use a pre-trained model (in this case, ResNet and Inception got the best performance) and only do a fine-tuning.

## Image Auotencoder

![Autoencoder-example](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Autoencoder-example.jpg)

Auotencoder-artwork notebook present a model to auotencode the artworks. The objective is to get a low dimensional representation of the image that catch the main features. There are two models:
- Denoising Autoencoder
- Sliced- Wasserstein Autoencoder

These representations keep some implicit relationships that can be exploted in artwork retrieval task (e.g curator work). As we can see in the image below, autoencoders bind artworks not only following style, otherwise the latent features allows to find underground relationship (some of them can be meaningful for curators). In the future, the objective is to get an intuition of that latent features in order to separate the meaningful from those that produce noise.

![T-sne distribution](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/tsne-analysis-denoisy.jpg)

## Artwork retrieval

![Artwork-retrieval-example](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Artwork-retrieval.jpg)

Artwork-retrieval notebook define the logic to search similar artworks using deep-autoencoders (defined on Auotencoder-artwork notebook). Basically, first we encode each image of the dataset it using a pre-trained encoder, obtaining a code matrix. Then, given an artwork, we encode it with the same encoder. Then, we look foward other similar artworks's codes. Finally, we take the top ten artworks associated to these codes.
Particularly, here we use a similarity measure that combines cosine distance and social influence. Basically, the idea is to adjust the cosine distance taking into account how far away are the artists in the influence social graph (this graph was built using DBpedia Ontology).

![Similarity-measure-explanation](https://github.com/ignaciogatti/art-deep-learning/blob/master/images/Similarity-measure.jpg)
