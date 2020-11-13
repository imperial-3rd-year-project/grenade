{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Grenade.Onnx.Iso where

import Grenade.Core.Layer
import Grenade.Onnx.OnnxLoadable

class Iso f where
  to   :: a -> f a
  from :: f a -> a

newtype Lift a = Lift { unlift :: a }

instance (UpdateLayer x, Iso f) => UpdateLayer (Lift (f x)) where
  type Gradient (Lift (f x)) = Gradient x
  type MomentumStore (Lift (f x)) = MomentumStore x
  runUpdate opt (Lift x) grad = Lift . to $ runUpdate opt (from x) grad
  runSettingsUpdate settings (Lift layer) = Lift . to $ runSettingsUpdate settings (from layer)
  reduceGradient grads = reduceGradient @x grads

instance (Layer x i o, Iso f) => Layer (Lift (f x)) i o where
  type Tape (Lift (f x)) i o = Tape x i o
  runForwards       = runForwards . from . unlift
  runBackwards      = runBackwards . from . unlift
  runBatchForwards  = runBatchForwards . from . unlift
  runBatchBackwards = runBatchBackwards . from . unlift

instance (RandomLayer x, Iso f) => RandomLayer (f x) where
  createRandomWith method gen = to <$> createRandomWith method gen

instance OnnxLoadable x => OnnxLoadable (Lift x) where
  loadOnnx tensors graph = (\(x, graph') -> (Lift x, graph')) <$> loadOnnx tensors graph
