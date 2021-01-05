{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TemplateHaskell       #-}

module Test.Grenade.Layers.Tanh where

import           Grenade.Layers.Tanh
import           Grenade.Core

import           Hedgehog

import           Numeric.LinearAlgebra.Static as H hiding ((===), create)
import           Numeric.LinearAlgebra.Data (toList)

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


tests :: IO Bool
tests = checkParallel $$(discover)

