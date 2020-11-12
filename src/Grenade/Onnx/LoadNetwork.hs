{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}

module Grenade.Onnx.LoadNetwork where

import Data.Singletons

import Grenade.Core.Network
import Grenade.Core.Layer
import Grenade.Onnx.OnnxLoadable

instance SingI i => OnnxLoadable (Network '[] '[i]) where
  loadOnnx _ graph = Just (NNil, graph)

instance (SingI i, SingI h, Layer x i h, OnnxLoadable x, OnnxLoadable (Network xs (h ': hs))) 
         => OnnxLoadable (Network (x ': xs) (i ': h ': hs)) where
  loadOnnx tensors graph = do
    (layer, graph')    <- loadOnnx tensors graph
    (network, graph'') <- loadOnnx tensors graph'
    return (layer :~> network, graph'')
