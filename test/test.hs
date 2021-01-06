{-# LANGUAGE CPP #-}
import           Control.Monad

import qualified Test.Grenade.Batch
import qualified Test.Grenade.Loss
import qualified Test.Grenade.Network

import qualified Test.Grenade.Layers.Add
import qualified Test.Grenade.Layers.BatchNorm
import qualified Test.Grenade.Layers.Convolution
import qualified Test.Grenade.Layers.Elu
import qualified Test.Grenade.Layers.FullyConnected
import qualified Test.Grenade.Layers.Logit
import qualified Test.Grenade.Layers.LeakyRelu
import qualified Test.Grenade.Layers.LRN
import qualified Test.Grenade.Layers.Mul
import qualified Test.Grenade.Layers.Nonlinear
import qualified Test.Grenade.Layers.PadCrop
import qualified Test.Grenade.Layers.Pooling
import qualified Test.Grenade.Layers.Relu
import qualified Test.Grenade.Layers.Reshape
import qualified Test.Grenade.Layers.Sinusoid
import qualified Test.Grenade.Layers.Softmax
import qualified Test.Grenade.Layers.Tanh
import qualified Test.Grenade.Layers.Transpose
import qualified Test.Grenade.Layers.Trivial

import qualified Test.Grenade.Layers.Internal.Convolution
import qualified Test.Grenade.Layers.Internal.Pooling
import qualified Test.Grenade.Layers.Internal.Transpose

import qualified Test.Grenade.Recurrent.Layers.LSTM

import qualified Test.Grenade.Sys.Networks
import qualified Test.Grenade.Sys.Training

import qualified Test.Grenade.Onnx.Graph
import qualified Test.Grenade.Onnx.Network

import           Grenade.Types

import           System.Exit
import           System.IO

main :: IO ()
main = do 
#if USE_FLOAT
  print "Testing with float type"
  print "Gradient checking is inaccute when using floats, to do gradient checking, build without the \"-use-float\" flag"
  disorderMain [
      Test.Grenade.Loss.tests
#else 
  print "Testing with double type"
  disorderMain [
      Test.Grenade.Network.tests
    , Test.Grenade.Loss.tests
#endif
    , Test.Grenade.Batch.tests
    , Test.Grenade.Layers.Add.tests
    , Test.Grenade.Layers.BatchNorm.tests
    , Test.Grenade.Layers.Convolution.tests
    , Test.Grenade.Layers.Elu.tests
    , Test.Grenade.Layers.FullyConnected.tests
    , Test.Grenade.Layers.Logit.tests
    , Test.Grenade.Layers.LeakyRelu.tests
    , Test.Grenade.Layers.LRN.tests
    , Test.Grenade.Layers.Mul.tests
    , Test.Grenade.Layers.Nonlinear.tests
    , Test.Grenade.Layers.PadCrop.tests
    , Test.Grenade.Layers.Pooling.tests
    , Test.Grenade.Layers.Relu.tests
    , Test.Grenade.Layers.Reshape.tests
    , Test.Grenade.Layers.Sinusoid.tests
    , Test.Grenade.Layers.Softmax.tests
    , Test.Grenade.Layers.Tanh.tests
    , Test.Grenade.Layers.Transpose.tests
    , Test.Grenade.Layers.Trivial.tests

    , Test.Grenade.Layers.Internal.Convolution.tests
    , Test.Grenade.Layers.Internal.Pooling.tests
    , Test.Grenade.Layers.Internal.Transpose.tests

    , Test.Grenade.Recurrent.Layers.LSTM.tests

    , Test.Grenade.Onnx.Network.tests
    , Test.Grenade.Onnx.Graph.tests
    
    , Test.Grenade.Sys.Networks.tests
    -- See note in Test/Grenade/Sys/Training.hs
    -- , Test.Grenade.Sys.Training.tests
    ]

disorderMain :: [IO Bool] -> IO ()
disorderMain tests = do
  lineBuffer
  rs <- sequence tests
  unless (and rs) exitFailure


lineBuffer :: IO ()
lineBuffer = do
  hSetBuffering stdout LineBuffering
  hSetBuffering stderr LineBuffering
