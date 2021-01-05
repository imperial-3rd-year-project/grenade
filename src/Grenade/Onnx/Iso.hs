{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-|
Module      : Grenade.Onnx.BypassLayer
Description : TODO Jason
-}

module Grenade.Onnx.Iso (Lift, Iso (..)) where

import Data.Proxy

import Grenade.Core.Layer
import Grenade.Onnx.OnnxOperator
import Grenade.Onnx.OnnxLoadable

import Control.DeepSeq

import Lens.Micro (over, _1)

-- | TODO Jason
class Iso f where
  to   :: a -> f a
  from :: f a -> a

-- | TODO Jason
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

instance (Show x, Iso f) => Show (Lift (f x)) where
  show (Lift x) = show (from x)

instance (NFData x, Iso f) => NFData (Lift (f x)) where
  rnf (Lift x) = rnf (from x)

instance (RandomLayer x, Iso f) => RandomLayer (f x) where
  createRandomWith method gen = to <$> createRandomWith method gen

instance OnnxOperator x => OnnxOperator (Lift x) where
  onnxOpTypeNames _ = onnxOpTypeNames (Proxy :: Proxy x)

instance OnnxLoadable x => OnnxLoadable (Lift x) where
  loadOnnx tensors graph = over _1 Lift <$> loadOnnx tensors graph
