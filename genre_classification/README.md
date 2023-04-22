# Genre Classification

## Movie classification with

### Pytorch

![image](https://user-images.githubusercontent.com/57833772/233807778-67b246dd-3aae-4b5b-815c-14560b01c610.png)

### Spacy

![image](https://user-images.githubusercontent.com/57833772/233807873-32c0118a-98bb-4f08-bdc4-6d9a5d197403.png)

## Problem: based on movie title and description assign its genre!

The tricky thing about the dataset is that it consists of 27 similar genre classes. Random guess model is 1/27 * 100% = 3.7% of correctly guessing the class (assuming the dataset is balanced). I approached classification in 3 ways:

### 1. Baseline.

Each "document" (title/description of the movie) is converted to a vector using Doc2Vec. Next, the title and description vectors are concanatenated and passed to Neural Networks. The baseline solution achieved 10% accuracy = 2.7 times better than random guess. In my opinion, this is pretty simple and efficient solution that doesn't require lots of computational power, and given that it's making more correct decisions than random guesses it could serve as a baseline.


### 2. LSTMs.

Each "document" (title/description of the movie) is tokenized and lematized using Spacy lematizer. Next, each word is embedded into vector space using GloVe (Word2Vec). Then (title, description) pair is processed with LSTMs implemented in PyTorch. The model's accuracy reaches 41% = 11 times better than random guess! :)

### 3. Spacy

Here the input data is converted to the Spacy format. Next, base_config and config files define the model and its hyperparameters. The model reached 10% accuracy. 
