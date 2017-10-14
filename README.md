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
This example trains a super-resolution network on the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), extracting patches from the 200 training images, and evaluating on crops of the 100 test images. 

## Example Usage:

### Train

`python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`

### Apply to low resolution images

`
