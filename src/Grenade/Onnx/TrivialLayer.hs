module Grenade.Onnx.TrivialLayer where

import Grenade.Onnx.OnnxLoadable
import Grenade.Onnx.Iso

class OnnxLoadableTrivial a where
  trivialLayer :: a

newtype LoadTrivial a = LoadTrivial a

instance Iso LoadTrivial where
  to = LoadTrivial
  from (LoadTrivial x) = x

instance OnnxLoadableTrivial x => OnnxLoadable (LoadTrivial x) where
  loadOnnx _ graph = Just (LoadTrivial trivialLayer, graph)
