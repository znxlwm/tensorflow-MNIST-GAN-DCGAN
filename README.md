# tensorflow-MNIST-GAN
Tensorflow implementation of Generative Adversarial Networks (GAN) for MNIST dataset.

## Resutls
* Generate using fixed noise (fixed_z_)

![Generation](MNIST_GAN_results/generation_animation.gif?raw=true)

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> Generated images </td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_GAN_results/MNIST_GAN_100.png'>
</tr>
</table>

* Training loss

![Loss](MNIST_GAN_results/MNIST_GAN_train_hist.png)

* Training time
    - Avg. per epoch: 4.97 sec
    - Total 100 epoch: 1255.92 sec

## Development Environment

* Windows 7
* GTX1080 ti
* cuda 8.0
* Python 3.5.3
* tensorflow-gpu 1.2.1
* numpy 1.13.1
* matplotlib 2.0.2
* imageio 2.2.0

## Reference

Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

(Full paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
