{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Relu
Description : Rectifying linear unit layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Relu (
    Relu (..)
  ) where

import           Control.DeepSeq                     (NFData (..))
import           Data.Maybe                          (fromJust)
import           Data.Proxy
import           Data.Serialize
import           GHC.Generics                        (Generic)
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static        as H

import           Grenade.Core
import           Grenade.Layers.Internal.Activations
import           Grenade.Onnx


-- | A rectifying linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
data Relu = Relu
  deriving (Generic, NFData, Show)

instance UpdateLayer Relu where
  type Gradient Relu = ()
  runUpdate _ _ _ = Relu
  reduceGradient _ = ()

instance RandomLayer Relu where
  createRandomWith _ _ = return Relu

instance Serialize Relu where
  put _ = return ()
  get = return Relu

instance (KnownNat i) => Layer Relu ('D1 i) ('D1 i) where
  type Tape Relu ('D1 i) ('D1 i) = S ('D1 i)

  runForwards _ (S1D y) =
    let i   = fromIntegral $ natVal (Proxy :: Proxy i)
        out = relu1d i (H.extract y)
    in  (S1D y, S1D . fromJust . H.create $ out)

  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (relu' y * dEdy))
    where
      relu' = H.dvmap (\a -> if a <= 0 then 0 else 1)

instance (KnownNat i, KnownNat j) => Layer Relu ('D2 i j) ('D2 i j) where
  type Tape Relu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards _ (S2D y) =
    let i   = fromIntegral $ natVal (Proxy :: Proxy i)
        j   = fromIntegral $ natVal (Proxy :: Proxy j)
        out = relu 1 i j (H.extract y)
    in  (S2D y, S2D . fromJust . H.create $ out)

  runBackwards _ (S2D y) (S2D dEdy) = ((), S2D (relu' y * dEdy))
    where
      relu' = H.dmmap (\a -> if a <= 0 then 0 else 1)

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Relu ('D3 i j k) ('D3 i j k) where

  type Tape Relu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards _ (S3D y) =
    let i   = fromIntegral $ natVal (Proxy :: Proxy i)
        j   = fromIntegral $ natVal (Proxy :: Proxy j)
        k   = fromIntegral $ natVal (Proxy :: Proxy k)
        out = relu k i j (H.extract y)
    in  (S3D y, S3D . fromJust . H.create $ out)

  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (relu' y * dEdy))
    where
      relu' = H.dmmap (\a -> if a <= 0 then 0 else 1)

instance OnnxOperator Relu where
  onnxOpTypeNames _ = ["Relu"]

instance OnnxLoadableActivation Relu where
  activationLayer = Relu
