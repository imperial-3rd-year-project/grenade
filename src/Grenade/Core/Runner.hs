{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-|
Module      : Grenade.Core.Runner
Description : Functions to perform training and backpropagation
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Core.Runner (
    train
  , validate
  , backPropagate
  , runNet
  ) where

import           Data.Singletons.Prelude

import qualified Numeric.LinearAlgebra.Static as SA

import           Grenade.Core.Network
import           Grenade.Core.Optimizer
import           Grenade.Core.Shape
import           Grenade.Core.TrainingTypes
import           Grenade.Types
import           Grenade.Utils.LinearAlgebra
import           Grenade.Core.Loss

-- | Perform reverse automatic differentiation on the network
--   for the current input and expected output.
backPropagate :: (SingI (Last shapes))
              => Network layers shapes
              -> S (Head shapes)
              -> S (Last shapes)
              -> LossFunction (S (Last shapes))
              -> Gradients layers
backPropagate network input target (LossFunction l) =
    let (tapes, output) = runNetwork network input
        (grads, _)      = runGradient network tapes (l output target)
    in  grads

validate :: Network layers shapes
         -> S (Head shapes)
         -> S (Last shapes)
         -> LossFunction (S (Last shapes))
         -> RealNum
validate network input target (LossFunction l) 
  = let (_, output) = runNetwork network input
    in case l output target of 
        (S1D x) -> sumV x

-- | Update a network with new weights after training with an instance.
train :: Optimizer opt
       -> Network layers shapes
       -> S (Head shapes)
       -> S (Last shapes)
       -> LossFunction (S (Last shapes))
       -> (Network layers shapes, RealNum)
train optimizer net input target (LossFunction l) =
    let (tapes, output) = runNetwork net input
        loss            = l output target
        (grads, _)      = runGradient net tapes loss
        net'            = applyUpdate optimizer net grads
    in case loss of 
        (S1D x) -> (net', sumV x)

-- | Run the network with input and return the given output.
runNet :: Network layers shapes -> S (Head shapes) -> S (Last shapes)
runNet net = snd . runNetwork net
