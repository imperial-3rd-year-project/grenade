{-# LANGUAGE ScopedTypeVariables  #-}

module Grenade.Onnx.TrivialLayer where

import Data.Proxy

import Grenade.Onnx.OnnxLoadable
import Grenade.Onnx.Iso

class OnnxOperator a => OnnxLoadableTrivial a where
  trivialLayer :: a

newtype LoadTrivial a = LoadTrivial a

instance Iso LoadTrivial where
  to = LoadTrivial
  from (LoadTrivial x) = x

instance OnnxOperator x => OnnxOperator (LoadTrivial x) where
  onnxOpTypeNames _ = onnxOpTypeNames (Proxy :: Proxy x)

instance OnnxLoadableTrivial x => OnnxLoadable (LoadTrivial x) where
  loadOnnxNode _ _ = Just (LoadTrivial trivialLayer)
