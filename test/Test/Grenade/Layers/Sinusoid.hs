{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TemplateHaskell       #-}

module Test.Grenade.Layers.Sinusoid where

import           Data.Serialize
import           Data.Either

import           Grenade.Layers.Sinusoid
import           Grenade.Core

import           Hedgehog

import           System.Random.MWC                        (create)

prop_sinusoid_is_serializable :: Property 
prop_sinusoid_is_serializable = withTests 1 $ property $ do
    gen <- evalIO create
    sinusoidLayer :: Sinusoid  <- evalIO $ createRandomWith UniformInit gen
    let enc = encode sinusoidLayer
        dec = decode enc :: Either String Sinusoid
    assert $ isRight dec


prop_can_update_sinusoid_and_use_in_batches :: Property 
prop_can_update_sinusoid_and_use_in_batches = property $ do
    gen <- evalIO create
    layer :: Sinusoid <- evalIO $ createRandomWith UniformInit gen
    runUpdate defSGD layer () `seq` success
    runUpdate defAdam layer () `seq` success
    reduceGradient @Sinusoid [()] `seq` success 
    
tests :: IO Bool
tests = checkParallel $$(discover)

