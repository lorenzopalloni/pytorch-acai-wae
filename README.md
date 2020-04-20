# Adversarially Constrained Autoencoder Interpolation (ACAI) using Wasserstein Autoencoder (WAE-MMD) and Wasserstein-Wasserstein Autoencoder (WWAE)

Results on MNIST dataset using ACAI + WWAE:
![ ](./presentation/figures/01-mnist-interpolations.jpg  "Results on MNIST with ACAI + WWAE")

Each row is made by reconstructing (a sample) of the linear convex combination of the latent representation between the first and the last element of the row.

## Training
In order to train ACAI with WWAE on MNIST dataset:

	$ python ./mnist/src/acwwaei.py

To start the training of ACAI using WAE on NESMDB dataset:

	$ python ./nesmdb/src/acwaei.py

## Reference

- [Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](https://arxiv.org/abs/1807.07543) 
- [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) 
- [Wasserstein-Wasserstein Auto-Encoders](https://arxiv.org/abs/1902.09323)
- [The NES Music Database: A multi-instrumental dataset with expressive performance attributes](https://arxiv.org/abs/1806.04278)
