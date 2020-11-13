{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE DefaultSignatures    #-}

module Grenade.Onnx.OnnxLoadable where

import           Control.Applicative
import           Control.Monad
import           Data.Proxy
import qualified Data.Map.Strict     as Map
import qualified Data.Text           as T
import           Lens.Micro

import           Grenade.Onnx.Onnx

import qualified Proto.Onnx          as P

class OnnxOperator a where
  onnxOpTypeNames :: Proxy a -> [T.Text]

class OnnxLoadable a where
  loadOnnx :: Map.Map T.Text P.TensorProto -> SPG 'S P.NodeProto -> Maybe (a, SPG 'S P.NodeProto)

  default loadOnnx :: OnnxOperator a => Map.Map T.Text P.TensorProto -> SPG 'S P.NodeProto -> Maybe (a, SPG 'S P.NodeProto)
  loadOnnx tensors (Node node) = (, Series []) <$> (node `hasType` (Proxy :: Proxy a) >> loadOnnxNode tensors node)
  loadOnnx tensors (Series ((Node node) : ns)) = fmap (Series ns <$) (loadOnnx tensors $ Node node)
  loadOnnx _ _ = Nothing

  loadOnnxNode :: Map.Map T.Text P.TensorProto -> P.NodeProto -> Maybe a
  loadOnnxNode tensors node = fst <$> loadOnnx tensors (Node node)

  {-# MINIMAL loadOnnx | loadOnnxNode #-}

hasType :: (Alternative m, OnnxOperator a) => P.NodeProto -> Proxy a -> m ()
hasType node a = guard $ node ^. #opType `elem` onnxOpTypeNames a
