{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Internal.Pooling where

import           Grenade.Layers.Internal.Pooling

import           Control.Monad

import           Numeric.LinearAlgebra                  hiding (konst,
                                                         uniformSample, (===))

import           Hedgehog
import qualified Hedgehog.Gen                           as Gen
import qualified Hedgehog.Range                         as Range

import qualified Test.Grenade.Layers.Internal.Reference as Reference
import           Test.Hedgehog.Compat

prop_poolForwards_poolBackwards_behaves_as_reference =
  let ok extent kernel = [stride | stride <- [1..extent], (extent - kernel) `mod` stride == 0]
      output extent kernel stride = (extent - kernel) `div` stride + 1
  in  property $ do
        height   <- forAll $ choose 2 100
        width    <- forAll $ choose 2 100
        kernel_h <- forAll $ choose 1 (height - 1)
        kernel_w <- forAll $ choose 1 (width - 1)
        stride_h <- forAll $ Gen.element (ok height kernel_h)
        stride_w <- forAll $ Gen.element (ok width kernel_w)
        input    <- forAll $ (height >< width) <$> Gen.list (Range.singleton $ height * width) (Gen.realFloat $ Range.linearFracFrom 0 (-100) 100)

        let outFast       = poolForward 1 height width kernel_h kernel_w stride_h stride_w input
        let retFast       = poolBackward 1 height width kernel_h kernel_w stride_h stride_w input outFast

        let outReference  = Reference.poolForward kernel_h kernel_w stride_h stride_w (output height kernel_h stride_h) (output width kernel_w stride_w) input
        let retReference  = Reference.poolBackward kernel_h kernel_w stride_h stride_w  input outReference

        outFast === outReference
        retFast === retReference

prop_same_pad_pool_behaves_as_reference_when_zero_pad =
  let output extent kernel_dim stride = (extent - kernel_dim) `div` stride + 1
      kernel i s = let x = ceiling ((fromIntegral i :: Double) / (fromIntegral s :: Double)) in i - (x - 1) * s
  in  property $ do
        height   <- forAll $ choose 2 100
        width    <- forAll $ choose 2 100

        stride_h <- forAll $ choose 1 (height - 1)
        stride_w <- forAll $ choose 1 (width - 1)

        let kernel_h = kernel height stride_h
            kernel_w = kernel width stride_w

        input    <- forAll $ (height >< width) <$> Gen.list (Range.singleton $ height * width) (Gen.realFloat $ Range.linearFracFrom 0 (-100) 100)

        guard $ output height kernel_h stride_h == (ceiling $ (fromIntegral height :: Double) / (fromIntegral stride_h :: Double))
        guard $ output width kernel_w stride_w  == (ceiling $ (fromIntegral width :: Double)  / (fromIntegral stride_w :: Double))

        let outFast       = validPadPoolForwards 1 height width kernel_h kernel_w stride_h stride_w 0 0 0 0 input
        let outReference  = poolForward 1 height width kernel_h kernel_w stride_h stride_w input

        assert $ norm_Inf (outFast - outReference) < 0.000001

prop_same_pad_pool_behaves_correctly_at_edges = withTests 1 $ property $ do
  let input           = (2 >< 2) [-0.01, -0.04, -0.02, -0.03]
      expected_output = (2 >< 2) [-0.01, -0.03, -0.02, -0.03]

      out = validPadPoolForwards 1 2 2 2 2 1 1 0 0 1 1 input

  assert $ norm_Inf (out - expected_output) < 0.000001

prop_same_pad_pool_behaves_correctly_at_edges_three_channels = withTests 1 $ property $ do
  let input           = (6 >< 2) [ 0.7, -0.9, -1.4, -0.1, -0.3, 0.5, 0.1, 0.2, -1.1, -0.7, 0.5, -0.7]
      expected_output = (6 >< 2) [ 0.7, -0.1, -0.1, -0.1, 0.5, 0.5, 0.2, 0.2, 0.5, -0.7, 0.5, -0.7]

      out = validPadPoolForwards 3 2 2 2 2 1 1 0 0 1 1 input

  assert $ norm_Inf (out - expected_output) < 0.000001

tests :: IO Bool
tests = checkParallel $$(discover)
