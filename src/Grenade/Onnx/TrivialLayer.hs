{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Grenade.Onnx.TrivialLayer where

import Grenade.Onnx.OnnxLoadable
import Grenade.Core.Layer

class OnnxLoadableTrivial a where
  trivialLayer :: a

newtype LoadTrivial a = LoadTrivial a

instance UpdateLayer x => UpdateLayer (LoadTrivial x) where
  type Gradient (LoadTrivial x) = Gradient x
  type MomentumStore (LoadTrivial x) = MomentumStore x
  runUpdate opt (LoadTrivial layer) grad = LoadTrivial $ runUpdate opt layer grad
  runSettingsUpdate settings (LoadTrivial layer) = LoadTrivial $ runSettingsUpdate settings layer
  reduceGradient grads = reduceGradient @x grads

instance Layer x i o => Layer (LoadTrivial x) i o where
  type Tape (LoadTrivial x) i o = Tape x i o
  runForwards (LoadTrivial x) s = runForwards x s
  runBackwards (LoadTrivial x) = runBackwards x
  runBatchForwards (LoadTrivial x) = runBatchForwards x
  runBatchBackwards (LoadTrivial x) = runBatchBackwards x

instance RandomLayer x => RandomLayer (LoadTrivial x) where
  createRandomWith method gen = LoadTrivial <$> createRandomWith method gen

instance OnnxLoadableTrivial x => OnnxLoadable (LoadTrivial x) where
  loadOnnx _ graph = Just (LoadTrivial trivialLayer, graph)
