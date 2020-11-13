{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Grenade.Onnx.ParallelLayer where

import           Data.Proxy
import           Data.Maybe (listToMaybe, catMaybes)

import Grenade.Onnx.OnnxLoadable
import Grenade.Onnx.Iso
import Grenade.Onnx.Onnx

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
      (, Series nodes) <$> (listToMaybe . catMaybes) (loadPair <$> [(x, y), (y, x)])
    where
      loadPair (x', y') = do
        (layerX, Series []) <- loadOnnx tensors x'
        (layerY, Series []) <- loadOnnx tensors y'
        return (to (mkParallelLayer layerX layerY))

  loadOnnx _ _ = Nothing
