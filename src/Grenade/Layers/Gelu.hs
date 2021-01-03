{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Gelu
Description : Gaussian Error Linear Unit (GELU)
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental

This module implements the Gaussian Error Linear Unit (GELU) activiation function. See

Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv:1606.08415 (2016).

Available at: https://arxiv.org/pdf/1606.08415.pdf

As in the paper we simply let μ = 0 and σ = 1. Futher, we use the simplified and thus fast representation: x * sigmoid (1.702 * x)

-}
module Grenade.Layers.Gelu (
    Gelu (..)
  ) where

import           Control.DeepSeq                (NFData (..))
import           Data.Serialize
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.Static   as LAS

import           Grenade.Core


-- | A Gaussion Error Linear Unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
--
--   Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv:1606.08415 (2016).
data Gelu = Gelu
  deriving (Generic, NFData, Show)

instance UpdateLayer Gelu where
  type Gradient Gelu = ()
  runUpdate _ _ _ = Gelu
  reduceGradient _ = ()

instance RandomLayer Gelu where
  createRandomWith _ _ = return Gelu

instance Serialize Gelu where
  put _ = return ()
  get = return Gelu

geluForwardFast :: Floating x => x -> x
geluForwardFast x = x / (e ** (-1.702 * x) + 1) -- = x * sigmoid (1.702 * x)
  where
    e = 2.71828

geluBackwardFast :: Floating x => x -> x
geluBackwardFast x = (e ** (1.702 * x) * (1 + e ** (1.702 * x) + 1.702 * x)) / (1 + e ** (1.702 * x)) ** 2
  where
    e = 2.71828

instance (KnownNat i) => Layer Gelu ('D1 i) ('D1 i) where
  type Tape Gelu ('D1 i) ('D1 i) = S ('D1 i)

  runForwards _ (S1D y) = (S1D y, S1D (gelu y))
    where
      gelu = LAS.dvmap geluForwardFast
  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (gelu' y * dEdy))
    where
      gelu' = LAS.dvmap geluBackwardFast

instance (KnownNat i, KnownNat j) => Layer Gelu ('D2 i j) ('D2 i j) where
  type Tape Gelu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards _ (S2D y) = (S2D y, S2D (gelu y))
    where
      gelu = LAS.dmmap geluForwardFast
  runBackwards _ (S2D y) (S2D dEdy) = ((), S2D (gelu' y * dEdy))
    where
      gelu' = LAS.dmmap geluBackwardFast

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Gelu ('D3 i j k) ('D3 i j k) where

  type Tape Gelu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards _ (S3D y) = (S3D y, S3D (gelu y))
    where
      gelu = LAS.dmmap geluForwardFast
  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (gelu' y * dEdy))
    where
      gelu' = LAS.dmmap geluBackwardFast
