# Train a RBM

To train a RBM, you need to use the rbms train script.

## RBM hyperparameters

- `--num_hiddens` Number of hidden nodes for the RBM. Setting it to $20$ or less allows to recover the exact log-likelihood of the model by enumerating on all hidden configurations.
- `--batch_size` Batch size, defaults to $2000$. Changing the batch size has an impact on the noise in the estimation of the positive term of the gradient. Setting it to a low value can lead to a very bad estimation and a bad training, but setting it too high can lead to an exact gradient, losing the benefits of the SGD (and remain trapped in a local minima for example).
- `--num_chains` Number of parallel chains, defaults to $2000$. Setting it to a much higher value than the batch size does not provide benefits, since it only impacts the estimation of the negative term of the gradient.
- `--gibbs_steps` Number of sampling steps performed at each gradient update. The $k$ in PCD-$k$.
- `--learning_rate` Learning rate. Defaults to $0.01$, setting a larger learning rate often leads to instability.
- `--num_updates` The training time is indexed on the number of gradient updates performed and not the number of epochs.
- `--beta` The inverse temperature to use during training (Defaults to $1$ and should not be changed)

## Save options

- `--filename` The path to the hdf5 archive to save the RBM during training. It will overwrite previously existing file.
- `--n_save` The number of machines to save during the training.
- `--spacing` Can be `exp` or `linear`, defaults to `exp`. When `exp` is selected, the time between the save of two models will increase exponentially. (It will look good in log-scale). When `linear` is selected, the time between the save of two models will be constant.
  Saving lots of models can quickly become the computational bottleneck, leading to long execution times.
- `--log` For now it is deprecated so you don't care about it.
- `--acc_ptt` Target acceptance rate. Defaults to $0.25$. Models will be saved when the acceptance rate between two consecutive models when sampling them using PTT drops below this threshold.
- `--acc_ll` Same as before but defaults to $0.75$. This allows to have two different schemes when saving models.

## PyTorch options

- `--device` The device on which to run the computations. Follows the PyTorch semantic so you can select which GPU to use with 'cuda:1' for example.
- `--dtype` The dtype of all the tensors. can be `int`, `double` or `float`. The default is `float` which corresponds to `torch.float32`.

## Example

The command I typically use to train a RBM on `MNIST-01` will be

```bash
rbms train -d ./path/to/MNIST.h5 --subset_labels 0 1 \
--filename output/rbm/MNIST01_from_scratch.h5  --num_updates 10000 \
--n_save 50 --spacing exp --num_hiddens 20 --batch_size 2000 --num_chains 2000 \
--learning_rate 0.01 --device cuda --dtype float
```

# Restore the training

If you want to continue the training of a RBM (be it one recovered from a RCM or a previously trained one), you can use the same scripts. The differences are that you should add the `--restore` flags. Also some arguments are not useful anymore and can be safely ignored:

- `--num_hiddens`
- `--batch_size`
- `--gibbs_steps`
- `--learning_rate`
- `--num_chains`

Finally the updates will be added to the same archive you provide as an input through `--filename`. If the `--restore` flag is set, then the file will **not** be overwritten.
