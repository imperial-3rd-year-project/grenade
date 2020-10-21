{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE TemplateHaskell       #-}

module Grenade.Core.Infer where

import Language.Haskell.TH
import Grenade.Core.Shape

--type MNIST
--  = Network
--    '[ Convolution 1 10 5 5 1 1, Pooling 2 2 2 2, Relu
--     , Convolution 10 16 5 5 1 1, Pooling 2 2 2 2, Reshape, Relu
--     , FullyConnected 256 80, Logit, FullyConnected 80 10, Logit]
--    '[ 'D2 28 28, 'D3 24 24 10, 'D3 12 12 10, 'D3 12 12 10
--     , 'D3 8 8 16, 'D3 4 4 16, 'D1 256, 'D1 256
--     , 'D1 80, 'D1 80, 'D1 10, 'D1 10]

class Shapeish tuple where
  toShape :: tuple -> Type

instance Integral a => Shapeish a where
  toShape x = AppT (PromotedT 'D1) (LitT $ NumTyLit $ toInteger x)
instance (Integral a, Integral b) => Shapeish (a, b) where
  toShape (x,y) = AppT (AppT (PromotedT 'D2) (LitT $ NumTyLit $ toInteger x))
                                             (LitT $ NumTyLit $ toInteger y)
instance (Integral a, Integral b, Integral c) => Shapeish (a, b, c) where
  toShape (x,y,z) = AppT (AppT (AppT (PromotedT 'D3) (LitT $ NumTyLit $ toInteger x))
                                                     (LitT $ NumTyLit $ toInteger y))
                                                     (LitT $ NumTyLit $ toInteger z)

layersToShapes :: Shapeish tuple => tuple -> [Name] -> Type
layersToShapes _ []     = PromotedNilT
layersToShapes t (n:ns) = AppT (AppT PromotedConsT $ toShape t) (listToDec undefined ns)
