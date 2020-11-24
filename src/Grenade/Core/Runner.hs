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
  , batchTrain
  , validate
  , backPropagate
  , runNet
  ) where

import           Data.Singletons.Prelude

import           Grenade.Utils.LinearAlgebra (nsum)
import           Grenade.Core.Network
import           Grenade.Core.Optimizer
import           Grenade.Core.Shape
import           Grenade.Core.TrainingTypes
import           Grenade.Types

-- | Perform reverse automatic differentiation on the network
--   for the current input and expected output.
backPropagate :: Network layers shapes
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
         -> (S (Last shapes) -> S (Last shapes) -> RealNum)
         -> RealNum
validate network input target l
  = let (_, output) = runNetwork network input
    in  l output target

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
    in (net', nsum loss)

batchTrain :: Optimizer opt
           -> Network layers shapes
           -> [S (Head shapes)]
           -> [S (Last shapes)]
           -> LossFunction (S (Last shapes))
           -> (Network layers shapes, RealNum)
batchTrain optimizer net inputs targets (LossFunction l) =
    let (tapes, outputs) = batchRunNetwork net inputs
        losses           = zipWith l outputs targets
        (grads, _)       = batchRunGradient net tapes losses
        net'             = applyUpdate optimizer net grads
        loss             = (sum $ map nsum losses) / (fromIntegral $ length losses)
    in (net', abs loss)

-- | Run the network with input and return the given output.
runNet :: Network layers shapes -> S (Head shapes) -> S (Last shapes)
runNet net = snd . runNetwork net
