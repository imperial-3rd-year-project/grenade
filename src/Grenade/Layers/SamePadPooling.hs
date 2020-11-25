{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}

module Grenade.Layers.SamePadPooling (
    SamePadPooling (..)
  ) where

import           Control.DeepSeq
import           Data.Function                   ((&))
import           Data.Kind                       (Type)
import           Data.Maybe
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits        hiding (natVal)
import           GHC.Generics
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static    (create, extract)

import           Grenade.Core
import           Grenade.Layers.Internal.Pooling
import           Grenade.Onnx

-- | A pooling layer for a neural network, for when auto_pad is SAME_UPPER or SAME_LOWER
--
--   Pads on the X and Y dimension of an image.
data SamePadPooling  :: Nat -> Nat -> Nat -> Nat -> Nat -> Nat -> Nat -> Nat -> Type where
  SamePadPooling :: SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom
  deriving (NFData, Generic)

instance Show (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) where
  show SamePadPooling = "SamePadPooling"

instance UpdateLayer (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) where
  type Gradient (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) = ()
  runUpdate _ x _ = x
  reduceGradient _ = ()

instance RandomLayer (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom)  where
  createRandomWith _ _ = return SamePadPooling

instance Serialize (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) where
  put _ = return ()
  get = return SamePadPooling

-- | A two dimentional image can be padded.
instance ( KnownNat padLeft
         , KnownNat padTop
         , KnownNat padRight
         , KnownNat padBottom
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , (padLeft + padRight) ~ ((outputColumns - 1) * strideColumns + kernelColumns - inputColumns)
         , (padTop + padBottom) ~ ((outputRows - 1) * strideRows + kernelRows - inputRows)
         ) => Layer (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns) where
  type Tape (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns)  = ()
  runForwards SamePadPooling (S2D input) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        padr  = fromIntegral $ natVal (Proxy :: Proxy padRight)
        padb  = fromIntegral $ natVal (Proxy :: Proxy padBottom)
        kr    = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        kc    = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sr    = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sc    = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        h     = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        w     = fromIntegral $ natVal (Proxy :: Proxy inputColumns)

        m     = extract input

        r     = validPadPoolForwards 1 h w kr kc sr sc padl padt padr padb m

    in  ((), S2D . fromJust . create $ r)
  runBackwards _ _ _ = error "backward pass for SamePadPooling not implemented"

-- | A two dimentional image can be padded.
instance ( KnownNat padLeft
         , KnownNat padTop
         , KnownNat padRight
         , KnownNat padBottom
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , KnownNat channels
         , KnownNat (inputRows * channels)
         , KnownNat (outputRows * channels)
         , (padLeft + padRight) ~ ((outputColumns - 1) * strideColumns + kernelColumns - inputColumns)
         , (padTop + padBottom) ~ ((outputRows - 1) * strideRows + kernelRows - inputRows)
         ) => Layer (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels) where
  type Tape (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels)  = ()
  runForwards SamePadPooling (S3D input) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        padr  = fromIntegral $ natVal (Proxy :: Proxy padRight)
        padb  = fromIntegral $ natVal (Proxy :: Proxy padBottom)
        kr    = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        kc    = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sr    = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sc    = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        h     = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        w     = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        c     = fromIntegral $ natVal (Proxy :: Proxy channels)

        m     = extract input

        r     = validPadPoolForwards c h w kr kc sr sc padl padt padr padb m
    in  ((), S3D . fromJust . create $ r)

  runBackwards _ _ _ = error "backward pass for SamePadPooling not implemented"

instance OnnxOperator (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) where
  onnxOpTypeNames _ = ["MaxPool"]

instance ( KnownNat padLeft
         , KnownNat padTop
         , KnownNat padRight
         , KnownNat padBottom
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat kernelRows
         , KnownNat kernelColumns
         ) => OnnxLoadable (SamePadPooling kernelRows kernelColumns strideRows strideColumns padLeft padTop padRight padBottom) where

  loadOnnxNode _ node = do
    node & hasSupportedDilations

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides")      strideShape

    -- todo: check that attribute is one of: SAME_UPPER or SAME_LOWER

    return SamePadPooling
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelColumns)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideColumns)]
