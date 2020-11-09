{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.BatchNorm where

import           Grenade.Core.Layer
import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Layers.BatchNormalisation
import           Grenade.Layers.Convolution
import           Grenade.Utils.ListStore

import           Numeric.LinearAlgebra             hiding (R, konst,
                                                    uniformSample, (===))
import           Numeric.LinearAlgebra.Data        as D hiding (R, (===))
import           Numeric.LinearAlgebra.Devel       as U
import           Numeric.LinearAlgebra.Static      as H hiding ((===))

import           Data.Maybe                        (fromJust)

import           Data.Proxy
import           GHC.TypeLits
import           Hedgehog
import qualified Hedgehog.Gen                      as Gen
import qualified Hedgehog.Range                    as Range

import           Debug.Trace
import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

convToList = D.toList . H.extract

batchNormLayer :: Bool -> BatchNorm 1 1 3 90
batchNormLayer training = 
  let gamma        = H.fromList [1, 1, 1]                :: R 3
      beta         = H.fromList [0, 0, 0]                :: R 3
      running_mean = H.fromList [-0.143, -0.135, -0.126] :: R 3
      running_var  = H.fromList [1.554, 1.550, 1.574]    :: R 3
  in  BatchNorm training (BatchNormParams gamma beta) running_mean running_var mkListStore

prop_batchnorm_initialisation = withTests 1 $ property $ do
  let bn :: BatchNorm 1 1 5 90                                            = initBatchNorm
      BatchNorm t (BatchNormParams gamma beta) running_mean running_var _ = bn

      zeroes  = replicate 5 0
      ones    = replicate 5 1

  t      === True
  ones   === convToList gamma
  zeroes === convToList beta
  zeroes === convToList running_mean
  ones   === convToList running_var

prop_batchnorm_1D_training_forward_pass = withTests 1 $ property $ do
  let bn = batchNormLayer True :: BatchNorm 1 1 3 90 

      batch = map (S1D . H.fromList) [[0.5, 0.1, 0.5], [0.3, 0.2, 0.3], [0.2, 0.01, 0.3]] :: [S ('D1 3)]

      (_, output) = runBatchForwards bn batch

      -- these reference values were produced using Python and tf.nn.batch_normalization
      refOutput = map (S1D . H.fromList) [[ 1.33626326, -0.04295011,  1.41413402], [-0.26725265,  1.24555325, -0.70706701], [-1.06901061, -1.20260314, -0.70706701]] :: [S ('D1 3)]

  assert . and $ zipWith allClose output refOutput

prop_batchnorm_1D_testing_forward_pass = withTests 1 $ property $ do
  let bn = batchNormLayer False :: BatchNorm 1 1 3 90 

      batch = map (S1D . H.fromList) [[0.5, 0.1, 0.5], [0.3, 0.2, 0.3], [0.2, 0.01, 0.3]] :: [S ('D1 3)]

      (_, output) = runBatchForwards bn batch

      -- these reference values were produced using Python and tf.nn.batch_normalization
      refOutput = map (S1D . H.fromList) [[0.51580474, 0.18875648, 0.49896701], [0.3553678, 0.26907839, 0.33955263], [0.27514934, 0.11646677, 0.33955263]] :: [S ('D1 3)]

  assert . and $ zipWith allClose output refOutput

tests :: IO Bool
tests = checkParallel $$(discover)
