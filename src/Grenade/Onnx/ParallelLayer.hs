{-# LANGUAGE OverloadedLabels       #-}
{-# LANGUAGE TupleSections          #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE UndecidableInstances   #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FlexibleContexts       #-}
{-# LANGUAGE GADTs                  #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-|
Module      : Grenade.Onnx.ParallelLayer
Description : Wrapper for automatically deriving OnnxLoadable instances for parallel layers like Concat.
-}

module Grenade.Onnx.ParallelLayer (LoadParallel, OnnxLoadableParallel (..)) where

import           Control.Applicative         ((<|>))
import           Data.List                   (foldl1')
import           Data.Either.Combinators     (rightToMaybe)
import           Data.Proxy

import           Grenade.Onnx.Graph
import           Grenade.Onnx.OnnxOperator
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.Utils
import           Grenade.Onnx.Iso

newtype LoadParallel a = LoadParallel a

instance Iso LoadParallel where
  to = LoadParallel
  from (LoadParallel x) = x

class OnnxOperator a => OnnxLoadableParallel a x y | a -> x, a -> y where
  mkParallelLayer :: x -> y -> a

instance OnnxOperator x => OnnxOperator (LoadParallel x) where
  onnxOpTypeNames _ = onnxOpTypeNames (Proxy :: Proxy x)

instance (OnnxLoadableParallel a x y, OnnxLoadable x, OnnxLoadable y)
         => OnnxLoadable (LoadParallel a) where
  loadOnnx tensors (Series (Parallel [x, y] : Node combineNode : nodes)) =
    combineNode `hasType` (Proxy :: Proxy a) >>
      case foldl1' (<|>) (loadPair <$> [(x, y), (y, x)]) of
        Just layer -> Right (layer, Just combineNode, Series nodes)
        Nothing    -> loadFailureAttr "Failed to load parallel layer" combineNode
    where
      loadPair (x', y') = do
        (layerX, _, Series []) <- rightToMaybe (loadOnnx tensors x')
        (layerY, _, Series []) <- rightToMaybe (loadOnnx tensors y')
        return (to (mkParallelLayer layerX layerY))

  loadOnnx _ (Series [Parallel _]) = loadFailureReason "Combine node missing"
  loadOnnx _ (Series []) = loadFailureReason "Graph unexpectedly ended"
  loadOnnx _ _ = loadFailureReason "Expecting parallel node"
