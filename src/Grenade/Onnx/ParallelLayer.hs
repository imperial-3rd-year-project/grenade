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
import qualified Data.Text           as T
import           Lens.Micro             ((^.))

import Grenade.Onnx.OnnxLoadable
import Grenade.Onnx.Iso
import Grenade.Onnx.Onnx

newtype LoadParallel a = LoadParallel a

instance Iso LoadParallel where
  to = LoadParallel
  from (LoadParallel x) = x

class OnnxLoadableParallel a x y | a -> x, a -> y where
  onnxOpTypeName :: Proxy a -> T.Text
  mkParallelLayer :: x -> y -> a

instance (OnnxLoadableParallel a x y, OnnxLoadable x, OnnxLoadable y) 
         => OnnxLoadable (LoadParallel a) where
  loadOnnx tensors (Series (Parallel [x, y] : combineNode : nodes))
    | isCombNode combineNode = 
      (, Series nodes) <$> (listToMaybe . catMaybes) (loadPair <$> [(x, y), (y, x)])
    where
      loadPair (x, y) = do
        (layerX, Series []) <- loadOnnx tensors x
        (layerY, Series []) <- loadOnnx tensors y
        return (to (mkParallelLayer layerX layerY))
      isCombNode (Node node) = node ^. #opType == onnxOpTypeName (Proxy :: Proxy a)
      isCombNode _ = False

  loadOnnx _ _ = Nothing
