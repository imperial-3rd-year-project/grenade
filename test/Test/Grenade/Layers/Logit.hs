{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TemplateHaskell       #-}

module Test.Grenade.Layers.Logit where

import           Data.Serialize
import           Data.Either

import           Grenade.Layers.Logit
import           Grenade.Core

import           Hedgehog

import           System.Random.MWC                        (create)

prop_logit_is_serializable :: Property 
prop_logit_is_serializable = withTests 1 $ property $ do
    gen <- evalIO create
    logitLayer :: Logit <- evalIO $ createRandomWith UniformInit gen
    let enc = encode logitLayer
        dec = decode enc :: Either String Logit
    assert $ isRight dec


prop_can_update_tanh_and_use_in_batches :: Property 
prop_can_update_tanh_and_use_in_batches = property $ do
    gen <- evalIO create
    layer :: Logit <- evalIO $ createRandomWith UniformInit gen
    runUpdate defSGD layer () `seq` success
    runUpdate defAdam layer () `seq` success
    reduceGradient @Logit [()] `seq` success 
    
tests :: IO Bool
tests = checkParallel $$(discover)

