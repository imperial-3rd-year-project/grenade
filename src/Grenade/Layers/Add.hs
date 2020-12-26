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

module Grenade.Layers.Add where

import           Control.DeepSeq              (NFData (..))

import           Data.Kind                    (Type)
import           Data.Maybe                   (fromJust)
import           Data.Proxy
import           Data.Serialize
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra        as LA
import           Numeric.LinearAlgebra.Static (R)
import qualified Numeric.LinearAlgebra.Static as H

import           Lens.Micro                   ((^.))

import           Grenade.Core
import           Grenade.Layers.Internal.Add
import           Grenade.Onnx

data Add :: Nat -- The number of channels of the bias
         -> Nat -- The number of rows of the bias
         -> Nat -- The number of columns of the bias
         -> Type where
  Add  :: ( KnownNat channels
          , KnownNat rows
          , KnownNat columns )
          => R (channels * rows * columns)
          -> Add channels rows columns

instance UpdateLayer (Add c h w) where
  type Gradient (Add c h w) = ()
  runUpdate _ x _  = x
  reduceGradient _ = ()

instance (KnownNat c, KnownNat h, KnownNat w ) => RandomLayer (Add c h w) where
  createRandomWith _ _ = pure initAdd

initAdd :: forall c h w. ( KnownNat c, KnownNat h, KnownNat w )
        => Add c h w
initAdd =
  let c'      = fromIntegral $ natVal (Proxy :: Proxy c)
      h'      = fromIntegral $ natVal (Proxy :: Proxy h)
      w'      = fromIntegral $ natVal (Proxy :: Proxy w)
      zeroes  = replicate (c' * h' * w') 0
      bias    = H.fromList zeroes   :: R (c * h * w)
  in Add bias

instance ( KnownNat c, KnownNat h, KnownNat w ) => Serialize (Add c h w) where
  put (Add bias) = putListOf put . LA.toList . H.extract $ bias
  get            = do
    bias <- maybe (fail "Vector of incorrect size") return . H.create . LA.fromList =<< getListOf get
    return $ Add bias

instance ( KnownNat i, KnownNat j, KnownNat k ) => Layer (Add k 1 1) ('D3 i j k) ('D3 i j k) where
  type Tape (Add k 1 1) ('D3 i j k) ('D3 i j k) = ()

  runForwards (Add b) (S3D m)
    = let c  = fromIntegral $ natVal (Proxy :: Proxy k)
          h  = fromIntegral $ natVal (Proxy :: Proxy i)
          w  = fromIntegral $ natVal (Proxy :: Proxy j)
          m' = H.extract m
          b' = H.extract b
          r  = addPerChannel c h w m' b'
      in  ((), S3D . fromJust . H.create $ r)

  runBackwards = undefined

instance OnnxOperator (Add c h w) where
  onnxOpTypeNames _ = ["Add"]

instance (KnownNat c, KnownNat h, KnownNat w) => OnnxLoadable (Add c h w) where
  loadOnnxNode inits node = case node ^. #input of
    [bias, _] -> do
      loadedBias <- readInitializerTensorIntoVector inits bias
      return $ Add loadedBias
    _ -> onnxIncorrectNumberOfInputs

instance NFData (Add c h w) where
  rnf (Add bias) = rnf bias
