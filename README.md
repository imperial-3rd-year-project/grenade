Grenade
=======

[Documentation](https://g206002126.pages.doc.ic.ac.uk/grenade/master/docs/index.html)

[Test Coverage](https://g206002126.pages.doc.ic.ac.uk/grenade/master/coverage-report/hpc_index.html)

Introduction
===========

This is a fork of [Manuel Schneckenreither's fork of the Grenade library](https://github.com/schnecki/grenade/tree/ee06cdde20ba89fc4d3c00a26dfe8b00796c7322). The original Grenade repo can be found at 
https://github.com/HuwCampbell/grenade. Some of the highlights of features we added are 

1. **ONNX network loading:** It is now possible to load the weights from ONNX networks. We currently support 
the popular networks Resnet, Yolo and SuperResolution! Adding new networks should be easy, simply define the 
structure of the network at type level, and then the ONNX interface will be able to load the network.
When running networks from ONNX, it is recommended that you build the library with the use-float flag 
if the network takes floats as inputs, since doubles will requires more memory space and will be slower.

2. **Training Interface:** There is now a training interface to abstract away the logic of supervised training 
on a neural network. Simply call the `fit` function and specify the type of the network and it will train
the network for you!

```haskell
trainMyNetwork :: IO MyNetwork
trainMyNetwork = fit trainingData validationData options epochs quadratic'
```

3. **Batch Training:** You can now perform mini-batch or batch training on a neural network. It is recommended to
do this via the training interface, but if you wish to define your own training functions, use the `reduceGradient` 
function for a layer to reduce a batch of gradients produced by back propogation into a single gradient 
that you can use for parameter tuning.  

4. **Loss Functions:** We have added a plethora of new training functions to Grenade. This allows users
to train a network using the most appropriate loss for the task, leading to better performance with 
neural networks. These can be used by themselves, or with the training interface. New loss functions 
include exponential, Hellinger, Kullback–Leibler, and Binary Crossentropy.  

5. **New Layers**: The layer zoo for Grenade has been massively expanded. It now supports the massively useful
batch normalization layer, which can benifit training in almost any network. Other new layers include Global Average Pooling,
Add, Mul, LRN, Convolutions with biases, and convolutions with padding. LeakyRelu has been changed so that the user can 
specify the alpha of the activation.

6. **Performance:** The performance of the layers has been greatly improved. Some layers have been rewritten in C
to quickly iterate over the matrices. Convolutions have been completely rewritten so that they use a more performant 
im2col trick that is approximately 25% faster than the older version.

7. **Improved Test Coverage:** Test coverage has been expanded massively, it gives users more confidence 
in the correctness of operations, and developers more security that code changes will not break the functionality.

Description
===========

```
First shalt thou take out the Holy Pin, then shalt thou count to three, no more, no less.
Three shall be the number thou shalt count, and the number of the counting shall be three.
Four shalt thou not count, neither count thou two, excepting that thou then proceed to three.
Five is right out.
```

💣 Machine learning which might blow up in your face 💣

Grenade is a composable, dependently typed, practical, and fast recurrent neural network library
for concise and precise specifications of complex networks in Haskell.

As an example, a network which can achieve ~1.5% error on MNIST can be
specified and initialised with random weights in a few lines of code with
```haskell
type MNIST
  = Network
    '[ Convolution 'WithoutBias 'NoPadding 1 10 5 5 1 1
     , Pooling 2 2 2 2, Relu
     , Convolution 'WithoutBias 'NoPadding 10 16 5 5 1 1
     , Pooling 2 2 2 2, Relu
     , Reshape
     , FullyConnected 256 80, Logit
     , FullyConnected 80 10, Logit
     ]
    '[ 'D2 28 28
     , 'D3 24 24 10, 'D3 12 12 10
     , 'D3 12 12 10
     , 'D3 8 8 16, 'D3 4 4 16
     , 'D1 256
     , 'D1 256, 'D1 80
     , 'D1 80, 'D1 10
     , 'D1 10
     ]

randomMnist :: MonadRandom m => m MNIST
randomMnist = randomNetwork
```

And that's it. Because the types are so rich, there's no specific term level code
required to construct this network; although it is of course possible and
easy to construct and deconstruct the networks and layers explicitly oneself.

Design
------

Networks in Grenade can be thought of as a heterogeneous lists of layers, where
their type includes not only the layers of the network, but also the shapes of
data that are passed between the layers.

The definition of a network is surprisingly simple:
```haskell
data Network :: [Type] -> [Shape] -> * where
    NNil  :: SingI i
          => Network '[] '[i]

    (:~>) :: (SingI i, SingI h, Layer x i h)
          => !x
          -> !(Network xs (h ': hs))
          -> Network (x ': xs) (i ': h ': hs)
```

The `Layer x i o` constraint ensures that the layer `x` can sensibly perform a
transformation between the input and output shapes `i` and `o`.

The lifted data kind `Shape` defines our 1, 2, and 3 dimension types, used to
declare what shape of data is passed between the layers.

In the MNIST example above, the input layer can be seen to be a two dimensional
(`D2`), image with 28 by 28 pixels. When the first *Convolution* layer runs, it
outputs a three dimensional (`D3`) 24x24x10 image. The last item in the list is
one dimensional (`D1`) with 10 values, representing the categories of the MNIST
data.

Usage
-----

To train a network, it is recommended to use the [`fit` function](https://g206002126.pages.doc.ic.ac.uk/grenade/master/docs/Grenade-Core-Training.html#v:fit). 

However, if you wish to define your own training function, you can use 
```haskell
runNetwork :: forall layers shapes.
              Network layers shapes -> S (Head shapes) -> (Tapes layers shapes, S (Last shapes))
```
which will produce the tapes needed to perform automatic differentiation for back propogation,
as well as the output of the network.

To perform back propagation, one can call the eponymous function
```haskell
backPropagate :: forall shapes layers.
                 Network layers shapes -> S (Head shapes) -> S (Last shapes) -> Gradients layers
```
which takes a network, appropriate input and target data, and returns the
back propagated gradients for the network. The shapes of the gradients are
appropriate for each layer, and may be trivial for layers like `Relu` which
have no learnable parameters.

The gradients however can always be applied, yielding a new (hopefully better)
layer with
```haskell
applyUpdate :: LearningParameters -> Network ls ss -> Gradients ls -> Network ls ss
```

Layers in Grenade are represented as Haskell classes, so creating one's own is
easy in downstream code. If the shapes of a network are not specified correctly
and a layer can not sensibly perform the operation between two shapes, then
it will result in a compile time error.

If you wish to perform minibatch gradient descent, you may want to use the function
```haskell
batchRunNetwork :: forall layers shapes.
                   Network layers shapes -> [S (Head shapes)] -> (BatchTapes layers shapes, [S (Last shapes)]) 
```
Batch tapes differ from normal tapes in that each layer will have a batch of tapes for backpropogation.

The next function of interest is (unsurprisingly named)
```haskell
batchRunGradient :: forall layers shapes.
               Network layers shapes -> BatchTapes layers shapes -> [S (Last shapes)] -> (Gradients layers, [S (Head shapes)])
```

Batch gradients are essential for layers like Batch Normalization whose back propogation is defined on the batch 
of gradients rather than a single gradient at a time.

Composition
-----------

Networks and Layers in Grenade are easily composed at the type level. As a `Network`
is an instance of `Layer`, one can use a trained Network as a small component in a
larger network easily. Furthermore, we provide 2 layers which are designed to run
layers in parallel and merge their output (either by concatenating them across one
dimension or summing by pointwise adding their activations). This allows one to
write any Network which can be expressed as a
[series parallel graph](https://en.wikipedia.org/wiki/Series-parallel_graph).

A residual network layer specification for instance could be written as
```haskell
type Residual net = Merge Trivial net
```
If the type `net` is an instance of `Layer`, then `Residual net` will be too. It will
run the network, while retaining its input by passing it through the `Trivial` layer,
and merge the original image with the output.

See the [Resnet18 definition](https://gitlab.doc.ic.ac.uk/g206002126/grenade/-/blob/master/src/Grenade/Networks/ResNet18.hs), 
which has been designed in such a way that is it the composition of many smaller networks.

Recurrent Neural Networks
-------------------------

If recurrent neural networks are more your style, you can try defining something
["unreasonably effective"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
with
```haskell
type Shakespeare
  = RecurrentNetwork
    '[ R (LSTM 40 80), R (LSTM 80 40), F (FullyConnected 40 40), F Logit]
    '[ 'D1 40, 'D1 80, 'D1 40, 'D1 40, 'D1 40 ]
```

Generative Adversarial Networks
-------------------------------

As Grenade is purely functional, one can compose its training functions in flexible
ways. [GAN-MNIST](https://github.com/HuwCampbell/grenade/blob/master/examples/main/gan-mnist.hs)
example displays an interesting, type safe way of writing a generative adversarial
training function in 10 lines of code.

Layer Zoo
---------

Grenade layers are normal haskell data types which are an instance of `Layer`, so
it's easy to build one's own downstream code. We do however provide a decent set
of layers, including convolution, deconvolution, pooling, pad, crop, logit, relu,
elu, tanh, and fully connected.

Build Instructions
------------------

### Installing Dependencies on Linux ###

This version of Grenade is most easily built with stack. You will also need the `lapack` and
`blas` libraries and development tools. To install these on Linux, run:

```
sudo apt-get install libgsl0-dev liblapack-dev libatlas-base-dev
```

You will also need to install protoc - this is used to read the protobuf ONNX files, to install
this, run the following (taken from the [official instructions](http://google.github.io/proto-lens/installing-protoc.html), but with the typo in the URL fixed).

```
PROTOC_ZIP=protoc-3.14.0-linux-x86_64.zip && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP && sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && rm -f $PROTOC_ZIP
```
### Installing Dependencies on macOS ###

You need `protobuf` to build Grenade on macOS. To install `protobuf` with Homebrew, run

```
brew install protobuf
```

To install from source, check out the [official instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md).

If you're using macOS 11 Big Sur, you might need to build with `cabal` and the latest version of ghc instead of `stack`.

Depending on your macOS version, you might also need to install `openblas` and add it to your path.

### Building Grenade for the First Time ###

Once you have all that, Grenade can be
build using:

```
stack build
```

and the tests run using:

```
stack test
```

You can easily switch from Double to Float vectors and matrices. Just provide the corresponding 
flag (use-float) when compiling (to all packages that):

```
stack clean && stack build --flag=grenade-examples:use-float --flag=grenade:use-float && stack bench
```
It is important to make sure you run `stack clean` before changing the flags, as not doing so may lead to segfaults
since it seems stack does not rebuild C files when the flags are changed.

This version of Grenade builds with GHC 8.10.

Thanks
------

First and foremost, we would like to thank Huw Campbell, without his original work on Grenade, 
this project simply would not exist. While adding features to this library, we have tried to 
keep with the original design philosophy as much as possible, it is a fantastic demonstration of 
the power of dependantly typed Haskell.

Secondly, we owe a big thank you to Manuel Schneckenreither. His work on Grenade showed us that it 
is possible to make Grenade a very competitive framework for neural networks. It is his work 
on Grenade that will build upon to create a more complete neural network library.

We would like to give a massive thanks to Nicolas Wu, who supervised us during this project.
He encouraged us to be optimistic in what we could create, but at the same time, was always there
to give us on advice on what was a realistic aim. Finally, we would like to thank his research group
at Imperial, who were always there to give us advice on our compile errors and seg faults - particularly
Jamie Willis, who taught us how to read core as we looked for performance issues.

Performance
-----------
Grenade is backed by hmatrix, BLAS, and LAPACK, with critical functions optimised
in C. Using the im2col trick popularised by Caffe, it should be sufficient for
many problems.

Being purely functional, it should also be easy to run batches in parallel, which
would be appropriate for larger networks, current examples however are single
threaded.

On a relatively modern i5 processor, it should be able to run an inference of Resnet18 in around 90ms.

Contributing
------------
Contributions are welcome.
