# Super-Resolution by using inverse convolutional network

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
  -=level               feature level, level = 0: pixel level
```
Super resolution task with a larger magnification (e.g., 4×). Test on MNIST dataset, LSUN Bedrooms
dataset, CelebA dataset

## Example Usage:

### Train

`python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`--level 2

### Apply to low resolution images

![1](https://user-images.githubusercontent.com/16787952/31579017-89b5507c-b0e1-11e7-83c9-c73dd7fd2d0f.png)
![2](https://user-images.githubusercontent.com/16787952/31579030-c4faa0ce-b0e1-11e7-9a9d-68d9fe43517b.png)
![3](https://user-images.githubusercontent.com/16787952/31579031-c5159de8-b0e1-11e7-8156-5069097f7880.png)
`
