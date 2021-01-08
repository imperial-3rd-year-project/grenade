{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-|

Module      : Grenade.Layers.Dropout
Description : Defines the Dropout Layer

Dropout is a regularization method used to prevent overfitting 
of the network to the training dataset. 
It works by randomly "dropping out", i.e. ignoring, outputs of 
a percentage of neurons.
This is strategy is effective in preventing overfitting as it 
forces a NN not to rely on a single feature for computing the 
output of a neural network
-}
module Grenade.Layers.Dropout (
    Dropout (..)
  , randomDropout
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive        (PrimBase, PrimState)
import           Data.Proxy
import           Data.Serialize
import           GHC.Generics                   hiding (R)
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static   hiding (Seed)
import           System.Random.MWC

import           Grenade.Core

-- Dropout layer help to reduce overfitting.
-- Idea here is that the vector is a shape of 1s and 0s, which we multiply the input by.
-- After backpropogation, we return a new matrix/vector, with different bits dropped out.
-- The provided argument is the proportion to drop in each training iteration (like 1% or
-- 5% would be reasonable).
data Dropout (pct :: Nat) =
  Dropout
    { dropoutActive :: Bool     -- ^ Add possibility to deactivate dropout
    , dropoutSeed   :: !Int     -- ^ Seed
    }
  deriving (Generic)

instance NFData (Dropout pct) where rnf (Dropout a s) = rnf a `seq` rnf s
instance Show (Dropout pct) where show (Dropout _ _) = "Dropout"
instance Serialize (Dropout pct) where
  put (Dropout act seed) = put act >> put seed
  get = Dropout <$> get <*> get

instance UpdateLayer (Dropout pct) where
  type Gradient (Dropout pct) = ()
  runUpdate _ (Dropout act seed) _ = Dropout act (seed+1)
  runSettingsUpdate set (Dropout _ seed) = Dropout (setDropoutActive set) seed
  reduceGradient _ = ()

instance RandomLayer (Dropout pct) where
  createRandomWith _ = randomDropout

randomDropout :: (PrimBase m) => Gen (PrimState m) -> m (Dropout pct)
randomDropout gen = Dropout True <$> uniform gen

instance (KnownNat pct, KnownNat i) => Layer (Dropout pct) ('D1 i) ('D1 i) where
  type Tape (Dropout pct) ('D1 i) ('D1 i) = R i
  runForwards (Dropout act seed) (S1D x)
    | not act = (v, S1D $ dvmap (rate *) x) -- multily with rate to normalise throughput
    | otherwise = (v, S1D $ v * x)
    where
      rate = (/100) $ fromIntegral $ max 0 $ min 100 $ natVal (Proxy :: Proxy pct)
      v = dvmap mask $ randomVector seed Uniform
      mask r
        | not act || r < rate = 1
        | otherwise = 0
  runBackwards (Dropout _ _) v (S1D x) = ((), S1D $ x * v)
