{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TemplateHaskell       #-}

module Test.Grenade.Layers.Elu where

import           Data.Serialize
import           Data.Either

import           Grenade.Layers.Elu
import           Grenade.Core

import           Hedgehog

import           System.Random.MWC                        (create)

prop_elu_is_serializable :: Property 
prop_elu_is_serializable = withTests 1 $ property $ do
    gen <- evalIO create
    eluLayer :: Elu  <- evalIO $ createRandomWith UniformInit gen
    let enc = encode eluLayer
        dec = decode enc :: Either String Elu
    assert $ isRight dec


prop_can_update_elu_and_use_in_batches :: Property 
prop_can_update_elu_and_use_in_batches = property $ do
    gen <- evalIO create
    layer :: Elu <- evalIO $ createRandomWith UniformInit gen
    runUpdate defSGD layer () `seq` success
    runUpdate defAdam layer () `seq` success
    reduceGradient @Elu [()] `seq` success 
    
tests :: IO Bool
tests = checkParallel $$(discover)

