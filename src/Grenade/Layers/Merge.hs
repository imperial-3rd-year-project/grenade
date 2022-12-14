{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE OverloadedStrings     #-}
{-|
Module      : Grenade.Core.Network
Description : Merging layer for parallel network composition
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Merge (
    Merge (..)
  ) where

import           Data.Serialize

import           Control.DeepSeq
import           Data.Singletons
import           GHC.Generics

#if MIN_VERSION_base(4,9,0)
import           Data.Kind       (Type)
#endif

import           Data.ProtoLens.Labels ()
import           Grenade.Core
import           Grenade.Onnx


-- | A Merging layer.
--
-- Similar to Concat layer, except sums the activations instead of creating a larger
-- shape.
data Merge :: Type -> Type -> Type where
  Merge :: !x -> !y -> Merge x y
  deriving (NFData, Generic)

instance (Show x, Show y) => Show (Merge x y) where
  show (Merge x y) = "Merge\n" ++ show x ++ "\n" ++ show y

-- | Run two layers in parallel, combining their outputs.
--   This just kind of "smooshes" the weights together.
instance (UpdateLayer x, UpdateLayer y) => UpdateLayer (Merge x y) where
  type Gradient (Merge x y) = (Gradient x, Gradient y)
  runUpdate opt (Merge x y) (x', y') = Merge (runUpdate opt x x') (runUpdate opt y y')
  reduceGradient grads = (reduceGradient @x xs, reduceGradient @y ys)
    where
      (xs, ys) = unzip grads

instance (RandomLayer x, RandomLayer y) => RandomLayer (Merge x y) where
  createRandomWith m gen = Merge <$> createRandomWith m gen <*> createRandomWith m gen

-- | Combine the outputs and the inputs, summing the output shape
instance (SingI i, SingI o, Layer x i o, Layer y i o) => Layer (Merge x y) i o where
  type Tape (Merge x y) i o = (Tape x i o, Tape y i o)

  runForwards (Merge x y) input =
    let (xT, xOut) = runForwards x input
        (yT, yOut) = runForwards y input
    in  ((xT, yT), xOut + yOut)

  runBackwards (Merge x y) (xTape, yTape) o =
    let (x', xB) = runBackwards x xTape o
        (y', yB) = runBackwards y yTape o
    in  ((x', y'), xB + yB)

instance (Serialize a, Serialize b) => Serialize (Merge a b) where
  put (Merge a b) = put a *> put b
  get = Merge <$> get <*> get

-------------------- OnnxLoadable instances --------------------

instance OnnxLoadableParallel (Merge x y) x y where
  mkParallelLayer = Merge

instance OnnxOperator (Merge x y) where
  onnxOpTypeNames _ = ["Add"]
