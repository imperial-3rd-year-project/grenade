{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Layers.GlobalAvgPool where

import           Control.DeepSeq                (NFData (..))
import           Data.Constraint                (Dict (..))
import           Data.Maybe                     (fromJust)
import           Data.Reflection                (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static   (L, R)
import qualified Numeric.LinearAlgebra.Static   as LAS

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Utils.LinearAlgebra

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

instance (KnownNat i) => Layer GlobalAvgPool ('D1 i) ('D1 1) where
  type Tape GlobalAvgPool ('D1 i) ('D1 1) = S ('D1 i)

  runForwards _ x = (x, S1D . listToVector $ [nsum x])

  runBackwards = undefined

instance (KnownNat i, KnownNat j) => Layer GlobalAvgPool ('D2 i j) ('D2 1 1) where
  type Tape GlobalAvgPool ('D2 i j) ('D2 1 1) = S ('D2 i j)

  runForwards _ x =
    let y' = [nsum x]
        y  = LAS.fromList $ y' :: L 1 1
    in  (x, S2D y)

  runBackwards = undefined

instance (KnownNat i, KnownNat j, KnownNat k) => Layer GlobalAvgPool ('D3 i j k) ('D3 1 1 k) where

  type Tape GlobalAvgPool ('D3 i j k) ('D3 1 1 k) = S ('D3 i j k)

  runForwards _ x =
    let cs = splitChannels x
        ys = map nsum cs
    in  (x, S3D . LAS.fromList $ ys )

  runBackwards = undefined
