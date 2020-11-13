{-# LANGUAGE ScopedTypeVariables  #-}

module Grenade.Onnx.BypassLayer where

import Data.Proxy

import Grenade.Onnx.OnnxLoadable
import Grenade.Onnx.Iso

class OnnxLoadableBypass a where
  bypassLayer :: a

newtype LoadBypass a = LoadBypass a

instance Iso LoadBypass where
  to = LoadBypass
  from (LoadBypass x) = x

instance OnnxOperator x => OnnxOperator (LoadBypass x) where
  onnxOpTypeNames _ = onnxOpTypeNames (Proxy :: Proxy x)

instance OnnxLoadableBypass x => OnnxLoadable (LoadBypass x) where
  loadOnnx _ graph = Just (LoadBypass bypassLayer, graph)
