# Mnist example in tensorflow

Foreword:
If there are no specific framework constraints, please use chainer or pytorch!!

The purpose of this repository is to introduce tensorflow notions through a Mnist example.
The sample codes on official repositories and such lack of clarity, or versatility (or both).

In mnist example version v1~, I tried to produce a parameterizable and yet practical code.

## Versions

### V0
This is a simple example written by a coworker that I based my examples on.

### V1
adds 
- device selection
- model as an `tf.keras.models.Model` object

### V2
adds 
- simple logging tool (inspired by Chainer)
- parameterizable lr descent

### V3
adds
- clean use of `tf.data.Dataset` and iterator
- evaluation on validation dataset each epoch
- pretty print of epoch progress with tqdm


## Environment
This was tested on tensorflow 1.9 with python 3.6 on this docker image: 
https://cloud.docker.com/u/guriido/repository/docker/guriido/tf

## Useful references:

#### On dataset creation and options
https://www.tensorflow.org/guide/performance/datasets

#### On datasets use with sessions
https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab

#### advanced tutorial
https://www.tensorflow.org/alpha/tutorials/eager