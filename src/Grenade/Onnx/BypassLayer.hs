{-# LANGUAGE ScopedTypeVariables #-}
{-|
Module      : Grenade.Onnx.BypassLayer
Description : Wrapper for automatically deriving OnnxLoadable instances for layers consuming no nodes.
-}

module Grenade.Onnx.BypassLayer (OnnxLoadableBypass (..), LoadBypass) where

import Data.Proxy

import Grenade.Onnx.OnnxOperator
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
  loadOnnx _ graph = Right (LoadBypass bypassLayer, Nothing, graph)
