{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE PolyKinds             #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Grenade.Layers.PaddedPooling (
    PaddedPooling
  ) where

import           Data.Function ((&))
import           Data.Proxy
import           GHC.TypeLits

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Layers.Pooling
import           Grenade.Layers.Pad
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.Graph
import           Grenade.Onnx.Iso
import           Grenade.Onnx.Utils

newtype PaddedPoolingIso a = PaddedPoolingIso {fromPaddedPoolingIso :: a}

instance Iso PaddedPoolingIso where
  to   = PaddedPoolingIso
  from = fromPaddedPoolingIso

type PaddedPooling' (input      :: Shape)
                    (output     :: Shape)
                    (kernelRows :: Nat) 
                    (kernelCols :: Nat) 
                    (strideRows  :: Nat) 
                    (strideCols  :: Nat) 
                    (padLeft    :: Nat) 
                    (padTop     :: Nat) 
                    (padRight   :: Nat) 
                    (padBottom  :: Nat)
  = Network
   '[ Pad padLeft padTop padRight padBottom , Pooling kernelRows kernelCols strideRows strideCols ] 
    (PaddedPoolingShapes padLeft padTop padRight padBottom input output)

type PaddedPooling input output kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom 
  = Lift (PaddedPoolingIso (PaddedPooling' input output kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom))

type family PaddedPoolingShapes (padLeft :: Nat) (padTop :: Nat) (padRight :: Nat) (padBottom :: Nat) (i :: Shape) (o :: Shape) :: [Shape] where
  PaddedPoolingShapes pl pt pr pb ('D2 rows cols) output
    = '[ 'D2 rows cols , 'D2 (pt + rows + pb) (pl + cols + pr) , output ]
  PaddedPoolingShapes pl pt pr pb ('D3 rows cols channels) output
    = '[ 'D3 rows cols channels, 'D3 (pt + rows + pb) (pl + cols + pr) channels, output ]

instance OnnxOperator (PaddedPoolingIso (Network
   '[ Pad padLeft padTop padRight padBottom , Pooling kernelRows kernelCols strideRows strideCols ] 
    ('[ 'D3 rows cols channels, 'D3 poolInputRows poolInputCols channels, 'D3 outputRows outputCols channels ]))) where
  onnxOpTypeNames _ = ["MaxPool"]


instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , KnownNat poolInputRows
         , KnownNat poolInputCols
         , (inputRows + padTop + padBottom) ~ poolInputRows
         , (inputCols + padLeft + padRight) ~ poolInputCols
         , strideRows * (outputRows - 1) <= (poolInputRows - kernelRows + 1) - 1
         , (poolInputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (poolInputCols - kernelCols + 1) - 1
         , (poolInputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat padLeft
         , KnownNat padRight
         , KnownNat padTop
         , KnownNat padBottom
         ) => OnnxLoadable (PaddedPoolingIso (Network
   '[ Pad padLeft padTop padRight padBottom , Pooling kernelRows kernelCols strideRows strideCols ] 
    ('[ 'D3 inputRows inputCols channels, 'D3 poolInputRows poolInputCols channels, 'D3 outputRows outputCols channels ]))) where

  loadOnnxNode _ node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides")      strideShape
    hasCorrectPadding node (Proxy :: Proxy padLeft) (Proxy :: Proxy padRight) (Proxy :: Proxy padTop) (Proxy :: Proxy padBottom)

    return $ PaddedPoolingIso (Pad :~> Pooling :~> NNil)
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]
