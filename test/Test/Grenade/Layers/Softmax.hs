{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TemplateHaskell       #-}

module Test.Grenade.Layers.Softmax where

import           Data.Serialize
import           Data.Either

import           Grenade.Layers.Softmax
import           Grenade.Core

import           Hedgehog

import           System.Random.MWC                        (create)

prop_softmax_is_serializable :: Property 
prop_softmax_is_serializable = withTests 1 $ property $ do
    gen <- evalIO create
    softmaxLayer :: Softmax  <- evalIO $ createRandomWith UniformInit gen
    let enc = encode softmaxLayer
        dec = decode enc :: Either String Softmax
    assert $ isRight dec


prop_can_update_softmax_and_use_in_batches :: Property 
prop_can_update_softmax_and_use_in_batches = property $ do
    gen <- evalIO create
    layer :: Softmax <- evalIO $ createRandomWith UniformInit gen
    runUpdate defSGD layer () `seq` success
    runUpdate defAdam layer () `seq` success
    reduceGradient @Softmax [()] `seq` success 
    
tests :: IO Bool
tests = checkParallel $$(discover)


