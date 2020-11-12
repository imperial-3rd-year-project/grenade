{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE UndecidableInstances #-}

module Grenade.Onnx.TrivialLayer where

import Grenade.Onnx.OnnxLoadable

class OnnxLoadableTrivial a where
  trivialLayer :: a

instance OnnxLoadableTrivial a => OnnxLoadable a where
  loadOnnx _ graph = Just (trivialLayer, graph)
