{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels      #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.LeakyRelu
Description : Rectifying linear unit layer
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.LeakyRelu (
    LeakyRelu (..)
  ) where

import           Control.DeepSeq                (NFData (..))
import           Data.Serialize
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.Static   as LAS

import           Grenade.Core
import           Grenade.Onnx
import           Grenade.Types


-- | A rectifying linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
newtype LeakyRelu = LeakyRelu RealNum
  deriving (Generic, NFData, Show)

instance UpdateLayer LeakyRelu where
  type Gradient LeakyRelu = ()
  runUpdate _ x _ = x
  reduceGradient _ = ()

instance RandomLayer LeakyRelu where
  createRandomWith _ _ = return $ LeakyRelu 0.01

instance Serialize LeakyRelu where
  put (LeakyRelu alpha) = put alpha
  get = LeakyRelu <$> get

instance (KnownNat i) => Layer LeakyRelu ('D1 i) ('D1 i) where
  type Tape LeakyRelu ('D1 i) ('D1 i) = S ('D1 i)

  runForwards (LeakyRelu alpha) (S1D y) = (S1D y, S1D (relu y))
    where
      relu = LAS.dvmap (\a -> if a < 0 then alpha * a else a)

  runBackwards (LeakyRelu alpha) (S1D y) (S1D dEdy) = ((), S1D (relu' y * dEdy))
    where
      relu' = LAS.dvmap (\a -> if a < 0 then alpha else 1)

instance (KnownNat i, KnownNat j) => Layer LeakyRelu ('D2 i j) ('D2 i j) where
  type Tape LeakyRelu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards (LeakyRelu alpha) (S2D y) = (S2D y, S2D (relu y))
    where
      relu = LAS.dmmap (\a -> if a < 0 then alpha * a else a)

  runBackwards (LeakyRelu alpha) (S2D y) (S2D dEdy) = ((), S2D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a < 0 then alpha else 1)

instance (KnownNat i, KnownNat j, KnownNat k) => Layer LeakyRelu ('D3 i j k) ('D3 i j k) where

  type Tape LeakyRelu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards (LeakyRelu alpha) (S3D y) = (S3D y, S3D (relu y))
    where
      relu = LAS.dmmap (\a -> if a < 0 then alpha * a else a)

  runBackwards (LeakyRelu alpha) (S3D y) (S3D dEdy) = ((), S3D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a < 0 then alpha else 1)

instance OnnxOperator LeakyRelu where
  onnxOpTypeNames _ = ["LeakyRelu"]

instance OnnxLoadable LeakyRelu where
  loadOnnxNode _ node = do
    alpha <- readDoubleAttribute "alpha" node

    return $ LeakyRelu alpha
