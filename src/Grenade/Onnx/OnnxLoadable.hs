{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE DefaultSignatures    #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE TypeOperators        #-}

module Grenade.Onnx.OnnxLoadable (OnnxLoadable (..)) where

import           Control.Applicative ((<|>))
import           Data.Bifunctor
import           Data.Singletons
import qualified Data.Map.Strict     as Map
import qualified Data.Text           as T
import           Lens.Micro

import           Grenade.Core.Network
import           Grenade.Core.Layer

import           Grenade.Onnx.OnnxLoadFailure
import           Grenade.Onnx.OnnxOperator
import           Grenade.Onnx.Graph
import           Grenade.Onnx.Utils

import qualified Proto.Onnx          as P

class OnnxLoadable a where
  loadOnnx :: Map.Map T.Text P.TensorProto -> SPG 'S P.NodeProto -> Either OnnxLoadFailure (a, Maybe P.NodeProto, SPG 'S P.NodeProto)

  default loadOnnx :: OnnxOperator a
                   => Map.Map T.Text P.TensorProto
                   -> SPG 'S P.NodeProto
                   -> Either OnnxLoadFailure (a, Maybe P.NodeProto, SPG 'S P.NodeProto)
  loadOnnx tensors (Node node) = bimap (over currentNode (<|> Just node)) (, Just node, Series []) (node `hasType` (Proxy :: Proxy a) >> loadOnnxNode tensors node)
  loadOnnx tensors (Series ((Node node) : ns)) = fmap (Series ns <$) (loadOnnx tensors $ Node node)
  loadOnnx _ _ = loadFailureExpecting "Unexpected Parallel node" Nothing

  loadOnnxNode :: Map.Map T.Text P.TensorProto -> P.NodeProto -> Either OnnxLoadFailure a
  loadOnnxNode tensors node = (^. _1) <$> loadOnnx tensors (Node node)

  {-# MINIMAL loadOnnx | loadOnnxNode #-}

instance SingI i => OnnxLoadable (Network '[] '[i]) where
  loadOnnx _ graph = Right (NNil, Nothing, graph)

instance (SingI i, SingI h, Layer x i h, OnnxLoadable x, OnnxLoadable (Network xs (h ': hs))) 
         => OnnxLoadable (Network (x ': xs) (i ': h ': hs)) where
  loadOnnx tensors graph = do
    (layer, lastSucc, graph') <- loadOnnx tensors graph
    bimap (over lastSuccessfulNode (<|> lastSucc)) (over _1 (layer :~>)) (loadOnnx tensors graph')
