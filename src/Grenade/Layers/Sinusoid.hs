{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Sinusoid
Description : Sinusoid nonlinear layer
Copyright   : (c) Manuel Schneckenreither, 2018
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Sinusoid
  ( Sinusoid(..)
  ) where

import           Control.DeepSeq                (NFData)
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics                   (Generic)

import           Grenade.Core

-- | A Sinusoid layer.
--   A layer which can act between any shape of the same dimension, performing a sin function.
data Sinusoid = Sinusoid
  deriving (NFData, Generic, Show)

instance UpdateLayer Sinusoid where
  type Gradient Sinusoid  = ()
  runUpdate _ _ _ = Sinusoid
  reduceGradient _ = ()

instance RandomLayer Sinusoid where
  createRandomWith _ _ = return Sinusoid

instance Serialize Sinusoid where
  put _ = return ()
  get = return Sinusoid

instance (a ~ b, SingI a) => Layer Sinusoid a b where
  type Tape Sinusoid a b = S a
  runForwards _ a = (a, sin a)
  runBackwards _ a g = ((), cos a * g)
