{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE GADTs                 #-}

module Test.Grenade.Onnx.FakeLayers where

import           Grenade

import           Control.DeepSeq
import           Data.Serialize
import qualified Data.Text            as T
import           Data.Proxy
import           Data.Kind            (Type)

import           GHC.Generics
import           GHC.TypeLits

data FakeParLayer :: Symbol -> Type -> Type -> Type where
  FakeParLayer :: !x -> !y -> FakeParLayer layer x y
  deriving (Generic, NFData, Eq, Show)

type FakeLayer (layer :: Symbol) = FakeParLayer layer Trivial Trivial

type ActFakeLayer (layer :: Symbol) = Lift (LoadActivation (FakeLayer layer))
type BypassFakeLayer (layer :: Symbol) = Lift (LoadBypass (FakeLayer layer))
type ParFakeLayer (layer :: Symbol) x y = Lift (LoadParallel (FakeParLayer layer x y))

type AlwaysFail (layer :: Symbol) = FakeParLayer layer Trivial Trivial

instance (Serialize x, Serialize y) => Serialize (FakeParLayer layer x y) where
  put (FakeParLayer x y) = put x *> put y
  get = FakeParLayer <$> get <*> get

instance UpdateLayer (FakeParLayer layer x y) where
  type Gradient (FakeParLayer layer x y) = ()
  runUpdate _ layer _ = layer
  reduceGradient _ = ()

instance (RandomLayer x, RandomLayer y) => RandomLayer (FakeParLayer layer x y) where
  createRandomWith m gen = FakeParLayer <$> createRandomWith m gen <*> createRandomWith m gen

instance (a ~ b) => Layer (FakeParLayer layer x y) a b where
  type Tape (FakeParLayer layer x y) a b = ()
  runForwards _ a = ((), a)
  runBackwards _ _ y = ((), y)

instance OnnxLoadableBypass (FakeParLayer layer Trivial Trivial) where
  bypassLayer = FakeParLayer Trivial Trivial

instance KnownSymbol layer => OnnxOperator (FakeParLayer layer x y) where
  onnxOpTypeNames _ = [T.pack (symbolVal (Proxy :: Proxy layer))]

instance KnownSymbol layer => OnnxLoadableActivation (FakeParLayer layer Trivial Trivial) where
  activationLayer = FakeParLayer Trivial Trivial

instance KnownSymbol layer => OnnxLoadableParallel (FakeParLayer layer x y) x y where
  mkParallelLayer = FakeParLayer

instance KnownSymbol layer => OnnxLoadable (AlwaysFail layer) where
  loadOnnxNode _ _ = loadFailureReason "AlwaysFail"
