{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TemplateHaskell       #-}

module Test.Grenade.Layers.Relu where

import           Data.Serialize
import           Data.Either

import           Grenade.Layers.Relu
import           Grenade.Core

import           Hedgehog

import           System.Random.MWC                        (create)

prop_relu_is_serializable :: Property 
prop_relu_is_serializable = withTests 1 $ property $ do
    gen <- evalIO create
    reluLayer :: Relu  <- evalIO $ createRandomWith UniformInit gen
    let enc = encode reluLayer
        dec = decode enc :: Either String Relu
    assert $ isRight dec


prop_can_update_relu_and_use_in_batches :: Property 
prop_can_update_relu_and_use_in_batches = property $ do
    gen <- evalIO create
    layer :: Relu <- evalIO $ createRandomWith UniformInit gen
    runUpdate defSGD layer () `seq` success
    runUpdate defAdam layer () `seq` success
    reduceGradient @Relu [()] `seq` success 
    
tests :: IO Bool
tests = checkParallel $$(discover)

