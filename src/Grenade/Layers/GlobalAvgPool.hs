{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE OverloadedStrings     #-}
{-|
Module      : Grenade.Layers.GlobalAvgPool
Description : Global Average Pooling
Maintainer  : Theo Charalambous
License     : BSD2
Stability   : experimental

The layer follows the implementation as introduced in the paper:
Min Lin, Qiang Chen, Shuicheng Yan. Network In Network
<https://arxiv.org/abs/1312.4400>
-}

module Grenade.Layers.GlobalAvgPool 
  (
  -- * Layer definition
    GlobalAvgPool (..)
  )
where

import           Control.DeepSeq                (NFData (..))
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics                   (Generic)
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static   as H

import           Grenade.Core
import           Grenade.Utils.LinearAlgebra

import           Grenade.Onnx

data GlobalAvgPool = GlobalAvgPool
  deriving (Generic, NFData, Show)

instance UpdateLayer GlobalAvgPool where
  type Gradient GlobalAvgPool = ()
  runUpdate _ _ _ = GlobalAvgPool
  reduceGradient _ = ()

instance RandomLayer GlobalAvgPool where
  createRandomWith _ _ = return GlobalAvgPool

instance Serialize GlobalAvgPool where
  put _ = return ()
  get   = return GlobalAvgPool

-- | Performing a global average pool on a 1d vector will take the average of the elements and
--   return a vector of size 1
instance (KnownNat i) => Layer GlobalAvgPool ('D1 i) ('D1 1) where
  type Tape GlobalAvgPool ('D1 i) ('D1 1) = S ('D1 i)

  runForwards _ x = 
    let n   = fromIntegral $ natVal (Proxy :: Proxy i)
        avg = nsum x / n
        vec = listToVector [avg]   
    in  (x, S1D vec)

  runBackwards = undefined

-- | Performing a global average pool on a 2d matrix will take the average of the elements
--   and return a matrix of size 1x1
instance (KnownNat i, KnownNat j) => Layer GlobalAvgPool ('D2 i j) ('D2 1 1) where
  type Tape GlobalAvgPool ('D2 i j) ('D2 1 1) = S ('D2 i j)

  runForwards _ x =
    let n   = fromIntegral $ natVal (Proxy :: Proxy i)
        m   = fromIntegral $ natVal (Proxy :: Proxy j)
        avg = nsum x / (n * m)
        mat = H.fromList [avg]
    in  (x, S2D mat)

  runBackwards = undefined

-- | Performing a global average pool on a 3d matrix with k channels will take the average of the elements
--   channel-wise and return a 3d matrix with k channels each of size 1x1
instance (KnownNat i, KnownNat j, KnownNat k) => Layer GlobalAvgPool ('D3 i j k) ('D3 1 1 k) where

  type Tape GlobalAvgPool ('D3 i j k) ('D3 1 1 k) = S ('D3 i j k)

  runForwards _ x =
    let n   = fromIntegral $ natVal (Proxy :: Proxy i)
        m   = fromIntegral $ natVal (Proxy :: Proxy j)
        cs  = splitChannels x
        ys  = map (\c -> nsum c / (n * m)) cs
        mat = H.fromList ys 
    in  (x, S3D mat)

  runBackwards = undefined

instance OnnxOperator GlobalAvgPool where
  onnxOpTypeNames _ = ["GlobalAveragePool"]

instance OnnxLoadableActivation GlobalAvgPool where
  activationLayer = GlobalAvgPool
