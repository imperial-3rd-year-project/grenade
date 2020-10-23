{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}

module Grenade.Core.Infer where

import Grenade.Core.Shape
import Data.Kind
import GHC.TypeLits
import Data.Singletons
import Data.Singletons.Prelude (Head, Last)

import Grenade.Core.Network
import Grenade.Layers

--type MNIST
--  = Network
--    '[ Convolution 1 10 5 5 1 1, Pooling 2 2 2 2, Relu
--     , Convolution 10 16 5 5 1 1, Pooling 2 2 2 2, Reshape, Relu
--     , FullyConnected 256 80, Logit, FullyConnected 80 10, Logit]
--    '[ 'D2 28 28, 'D3 24 24 10, 'D3 12 12 10, 'D3 12 12 10
--     , 'D3 8 8 16, 'D3 4 4 16, 'D1 256, 'D1 256
--     , 'D1 80, 'D1 80, 'D1 10, 'D1 10]
{--
data Test a = Yes a | No Int

type instance Transformable ('D1 x) (Test Int) = 'D1 (x + 1)
type instance Transformable ('D1 x) (Test Char) = 'D1 (2 * x)

type instance Transformable s Tanh  = s
type instance Transformable s Relu  = s
type instance Transformable s Logit = s
type instance Transformable ('D1 i) (FullyConnected i o) = ('D1 o)
type instance Transformable ('D1 i) (Network _ (('D1 i) ': shapes)) = Last shapes

type IntTest = Test Int

type MyShapes = Create ('D1 2) '[IntTest, Test Char]

type MyLayers = '[ FullyConnected 2 40, Tanh, FullyConnected 40 10, Relu, FullyConnected 10 1, Logit ]

type BrunosLayers = '[ FFNet, Tanh ]

type FFNet = Network MyLayers (Create ('D1 2) MyLayers)  
type NewNet = Network BrunosLayers (Create ('D1 2) BrunosLayers)

--'[ 'D1 2, 'D1 40, 'D1 40, 'D1 10, 'D1 10, 'D1 1, 'D1 1]

--}