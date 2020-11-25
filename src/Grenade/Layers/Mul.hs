{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels      #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}

module Grenade.Layers.Mul where

import           Data.Kind                    (Type)
import           Data.Proxy
import           Data.Serialize
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra        as LA
import           Numeric.LinearAlgebra.Static (R)
import qualified Numeric.LinearAlgebra.Static as H

import           Lens.Micro                   ((^.))

import           Grenade.Core
import           Grenade.Onnx

data Mul :: Nat -- The number of channels of the bias
         -> Nat -- The number of rows of the bias
         -> Nat -- The number of columns of the bias
         -> Type where
  Mul  :: ( KnownNat channels
          , KnownNat rows
          , KnownNat columns )
          => R (channels * rows * columns)
          -> Mul channels rows columns

instance UpdateLayer (Mul c h w) where
  type Gradient (Mul c h w) = ()
  runUpdate _ x _  = x
  reduceGradient _ = ()

instance (KnownNat c, KnownNat h, KnownNat w ) => RandomLayer (Mul c h w) where
  createRandomWith _ _ = pure initMul

initMul :: forall c h w. ( KnownNat c, KnownNat h, KnownNat w )
        => Mul c h w
initMul =
  let c'    = fromIntegral $ natVal (Proxy :: Proxy c)
      h'    = fromIntegral $ natVal (Proxy :: Proxy h)
      w'    = fromIntegral $ natVal (Proxy :: Proxy w)
      ones  = replicate (c' * h' * w') 1
      bias  = H.fromList ones :: R (c * h * w)
  in Mul bias

instance ( KnownNat c, KnownNat h, KnownNat w ) => Serialize (Mul c h w) where
  put (Mul bias) = putListOf put . LA.toList . H.extract $ bias
  get            = do
    bias <- maybe (fail "Vector of incorrect size") return . H.create . LA.fromList =<< getListOf get
    return $ Mul bias

instance ( KnownNat i, KnownNat j, KnownNat k ) => Layer (Mul 1 1 1) ('D3 i j k) ('D3 i j k) where
  type Tape (Mul 1 1 1) ('D3 i j k) ('D3 i j k) = ()

  runForwards (Mul b) (S3D m)
    = let s  = H.extract b LA.! 0
      in  ((), S3D $  H.dmmap (s *) m)

  runBackwards = undefined

instance OnnxOperator (Mul c h w) where
  onnxOpTypeNames _ = ["Mul"]

instance (KnownNat c, KnownNat h, KnownNat w) => OnnxLoadable (Mul c h w) where
  loadOnnxNode inits node = case node ^. #input of
    [_, scale] -> do
      loadedScale <- readInitializerVector inits scale

      return $ Mul loadedScale
    _               -> onnxIncorrectNumberOfInputs

