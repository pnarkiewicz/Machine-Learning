# Machine-Learning

In this repository I include my most exciting projects related to Machine Learning!

The projects include implementations of:
- Multi-Head Attention Mechanism with caching (text generation) - https://arxiv.org/pdf/1706.03762.pdf
- Reinforcement Learning Agent - BFF DQN
- Generative Adversarial Networks (image generation)
- Variational Autoencoders (image generation)
- Attack on classification network - MobileNet
- Visual Anomaly Detection - https://arxiv.org/pdf/2011.08785.pdf
- Classification of Genre News using Neural Networks, LSTM, Spacy

## 1. ![Multi-Head Attention Mechanism](https://github.com/pnarkiewicz/Machine-Learning/blob/main/Deep%20Neural%20Networks%20-%20Notebooks/Transformer-Attention.ipynb)

The implementation of Multi-Head Attention is a decoder-only version. The notebook includes the implementation of the transformer architecture, its training and inference which at times might be tedious and inefficient. Hence, two solutionswere implemented:
- inference with cache boosting speed of generation
- nucleus sampling increasing the accuracy by keeping top n sequences through generation
Data used for training and testing purposes was generated synthetically.

## 2. Reinforcement Learning - Bigger Better Faster DQN

The RL Agent trains Deep-Q Network how to correctly land a spaceship on a moon. The BFF improvmenets are based on https://arxiv.org/pdf/2305.19452.pdf and include:
- n-step buffering - is supposed to encourage DQN to look further into the future
- resets - DQNs learn often incorrectly on random data, hence, once we have less random dataset we reset DQNs but keep the better data
- annealing - decreases the discount hyperparameter

## 3. Generative Adversarial Networks

Here we train Generator and Discriminator to generate most realistic digits images.

## 4. Variational Autoencoder

Variational Autoencoder tries to generate digits images as GANs and can be used for comparison to see that GANs perform typically better.

## 5. Attack on MobileNet

The task is to compute a top strip of goldfish image so that the pre-trained MobileNet thinks its shark. I implemented Neural Network where the weights are the top strip that we're supposed to derive.

## 6. Visual Anomaly Detection

The code is based on a paper - https://arxiv.org/pdf/2011.08785.pdf which describes the PADIM architecture. It's pretty lightweight and up-to ~2021 was state-of-the-art in anomaly detection which is pretty surprising as it "just" computes mean and covariance matrices.

## 7. Genere Classification 

Here I implemented Neural Networks, LSTM and Spacy classification model to test which of the solutions is the best. The task was challenging as there are 27 similar genres to choose from. The baseline Neural Network solution achieved 11% accurcy (likewise Spacy model). LSTMs turned out to be the most accurate with accuracy reaching 41%!



