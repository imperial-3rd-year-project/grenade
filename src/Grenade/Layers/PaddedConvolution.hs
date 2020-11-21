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
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE OverloadedLabels      #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Grenade.Layers.PaddedConvolution (
    PaddedConvolution
  ) where

import           Data.Function           ((&))
import           Data.Proxy
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static (tr)

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Layers.Convolution
import           Grenade.Layers.Pad
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.Graph
import           Grenade.Onnx.Utils
import           Grenade.Utils.ListStore

import           Grenade.Onnx.Iso
import           Lens.Micro ((^.))

newtype PaddedConvolutionIso a = PaddedConvolutionIso {fromPaddedConvolutionIso :: a}

instance Iso PaddedConvolutionIso where
  to   = PaddedConvolutionIso
  from = fromPaddedConvolutionIso

type PaddedConvolution' (input      :: Shape)
                       (output     :: Shape)
                       (channels   :: Nat) 
                       (filters    :: Nat)
                       (kernelRows :: Nat) 
                       (kernelCols :: Nat) 
                       (strideRows  :: Nat) 
                       (strideCols  :: Nat) 
                       (padLeft    :: Nat) 
                       (padTop     :: Nat) 
                       (padRight   :: Nat) 
                       (padBottom  :: Nat)
  = Network
   '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] 
    (PaddedConvolutionShapes padLeft padTop padRight padBottom input output)

type PaddedConvolution input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom = Lift (PaddedConvolutionIso (PaddedConvolution' input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom))

type family PaddedConvolutionShapes (padLeft :: Nat) (padTop :: Nat) (padRight :: Nat) (padBottom :: Nat) (i :: Shape) (o :: Shape) :: [Shape] where
  PaddedConvolutionShapes pl pt pr pb ('D2 rows cols) output
    = '[ 'D2 rows cols , 'D2 (pt + rows + pb) (pl + cols + pr) , output ]
  PaddedConvolutionShapes pl pt pr pb ('D3 rows cols channels) output
    = '[ 'D3 rows cols channels, 'D3 (pt + rows + pb) (pl + cols + pr) channels, output ]

instance OnnxOperator (PaddedConvolutionIso (Network
   '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] 
    '[ 'D3 rows cols channels, 'D3 convInputRows convInputCols channels, 'D3 outputRows outputCols filters ])) where
  onnxOpTypeNames _ = ["Conv"]


instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , KnownNat convInputRows
         , KnownNat convInputCols
         , (inputRows + padTop + padBottom) ~ convInputRows
         , (inputCols + padLeft + padRight) ~ convInputCols
         , strideRows * (outputRows - 1) <= (convInputRows - kernelRows + 1) - 1
         , (convInputRows - kernelRows + 1) <= outputRows * strideRows
         , strideCols * (outputCols - 1) <= (convInputCols - kernelCols + 1) - 1
         , (convInputCols - kernelCols + 1) <= outputCols * strideCols
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * filters)
         , KnownNat padLeft
         , KnownNat padRight
         , KnownNat padTop
         , KnownNat padBottom
         ) => OnnxLoadable (PaddedConvolutionIso (Network
   '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] 
   '[ 'D3 inputRows inputCols channels, 'D3 convInputRows convInputCols channels, 'D3 outputRows outputCols filters ])) where

  loadOnnxNode inits node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernelShape") kernelShape
    (node `hasMatchingShape` "strides"    ) strideShape
    (node `hasCorrectPadding`) (Proxy :: Proxy padLeft) (Proxy :: Proxy padRight) (Proxy :: Proxy padTop) (Proxy :: Proxy padBottom)

    case node ^. #input of
      [_, w] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        return $ PaddedConvolutionIso (Pad :~> Convolution filterWeights mkListStore :~> NNil)
      _ -> Nothing
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]
