{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}

{-|
Module      : Grenade.Core.Softmax
Description : Softmax loss layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Softmax (
    Softmax (..)
  , softmax
  , softmax'
  ) where

import           Data.Serialize

import           Control.DeepSeq                (NFData (..))
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static   as LAS

import           Grenade.Core

-- | A Softmax layer
--
--   This layer is like a logit layer, but normalises
--   a set of matricies to be probabilities.
--
--   One can use this layer as the last layer in a network
--   if they need normalised probabilities.
data Softmax = Softmax
  deriving (Show, Generic, NFData)

instance UpdateLayer Softmax where
  type Gradient Softmax = ()
  runUpdate _ _ _ = Softmax
  reduceGradient _ = ()

instance RandomLayer Softmax where
  createRandomWith _ _ = return Softmax

instance ( KnownNat i ) => Layer Softmax ('D1 i) ('D1 i) where
  type Tape Softmax ('D1 i) ('D1 i) = S ('D1 i)

  runForwards _ (S1D y) = (S1D y, S1D (softmax y))
  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (softmax' y dEdy))

instance Serialize Softmax where
  put _ = return ()
  get = return Softmax

softmax :: KnownNat i => LAS.R i -> LAS.R i
softmax xs =
  let xs' = LAS.dvmap exp xs
      s   = LAS.dot xs' 1
  in  LAS.dvmap (/ s) xs'

softmax' :: KnownNat i => LAS.R i -> LAS.R i -> LAS.R i
softmax' x grad =
  let yTy = outer sm sm
      d   = diag sm
      g   = d - yTy
  in  g #> grad
    where
  sm = softmax x
