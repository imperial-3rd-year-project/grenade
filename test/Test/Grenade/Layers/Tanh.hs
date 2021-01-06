{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TemplateHaskell       #-}

module Test.Grenade.Layers.Tanh where

import           Data.Serialize
import           Data.Either

import           Grenade.Layers.Tanh
import           Grenade.Core

import           Hedgehog

import           Numeric.LinearAlgebra.Static as H hiding ((===), create)
import           Numeric.LinearAlgebra.Data (toList)

import           System.Random.MWC                        (create)

failLargeDiff :: (MonadTest m) => Double -> m ()
failLargeDiff x = diff x (<) 0.001

prop_run_forward_applies_tanh_to_1d_case :: Property 
prop_run_forward_applies_tanh_to_1d_case = property $ do
  let input       = S1D $ H.fromList [-1, -0.5, 0, 0.5, 1] :: S ('D1 5)
      expectedOut = H.fromList [-0.7616, -0.4621, 0, 0.4621, 0.7616] :: R 5

  let layer  = Tanh
      (_, output) = runForwards layer input
      S1D output' = output
      result      = ((abs <$>) . toList . unwrap) (output' - expectedOut) 

  sequence_ (failLargeDiff <$> result)

prop_tanh_is_serializable :: Property 
prop_tanh_is_serializable = withTests 1 $ property $ do
    gen <- evalIO create
    tanhLayer :: Tanh  <- evalIO $ createRandomWith UniformInit gen
    let enc = encode tanhLayer
        dec = decode enc :: Either String Tanh
    assert $ isRight dec


prop_can_update_tanh_and_use_in_batches :: Property 
prop_can_update_tanh_and_use_in_batches = property $ do
    gen <- evalIO create
    layer :: Tanh <- evalIO $ createRandomWith UniformInit gen
    runUpdate defSGD layer () `seq` success
    runUpdate defAdam layer () `seq` success
    reduceGradient @Tanh [()] `seq` success 
    
tests :: IO Bool
tests = checkParallel $$(discover)

