{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE UndecidableInstances #-}

module Grenade.Onnx.OnnxLoadable where


import           Control.Applicative
import           Control.Monad
import           Data.Proxy
import           Data.Maybe (listToMaybe, catMaybes)
import qualified Data.Map.Strict     as Map
import qualified Data.Text           as T
import           Lens.Micro

import           Grenade.Onnx.Onnx

import qualified Proto.Onnx          as P

class OnnxLoadable a where
  loadOnnx :: Map.Map T.Text P.TensorProto -> SPG 'S P.NodeProto -> Maybe (a, SPG 'S P.NodeProto)

class OnnxLoadableParallel a where
  onnxOpTypeName :: Proxy a -> T.Text
  mkParallelLayer :: x -> y -> a x y

instance (OnnxLoadableParallel a, OnnxLoadable x, OnnxLoadable y) => OnnxLoadable (a x y) where
  loadOnnx tensors (Series (Parallel [x, y] : combineNode : nodes))
    | isCombNode combineNode = 
      (, Series nodes) <$> (listToMaybe . catMaybes) (loadPair <$> [(x, y), (y, x)])
    where
      loadPair (x, y) = do
        (layerX, Series []) <- loadOnnx tensors x
        (layerY, Series []) <- loadOnnx tensors y
        return (mkParallelLayer layerX layerY)
      isCombNode (Node node) = node ^. #opType == onnxOpTypeName (Proxy :: Proxy a)
      isCombNode _ = False

  loadOnnx _ _ = Nothing

hasType :: Alternative m => P.NodeProto -> T.Text -> m ()
hasType node typeString = guard $ typeString == (node ^. #opType)
