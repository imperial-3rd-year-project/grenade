{-# LANGUAGE ScopedTypeVariables  #-}

module Grenade.Onnx.ActivationLayer where

import Data.Proxy

import Grenade.Onnx.OnnxLoadable
import Grenade.Onnx.Iso

class OnnxOperator a => OnnxLoadableActivation a where
  activationLayer :: a

newtype LoadActivation a = LoadActivation a

instance Iso LoadActivation where
  to = LoadActivation
  from (LoadActivation x) = x

instance OnnxOperator x => OnnxOperator (LoadActivation x) where
  onnxOpTypeNames _ = onnxOpTypeNames (Proxy :: Proxy x)

instance OnnxLoadableActivation x => OnnxLoadable (LoadActivation x) where
  loadOnnxNode _ _ = Just (LoadActivation activationLayer)