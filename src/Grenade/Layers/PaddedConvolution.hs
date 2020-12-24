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

import           Data.Function                ((&))
import           Data.Proxy
import           GHC.TypeLits
import           Lens.Micro                   ((^.))
import           Numeric.LinearAlgebra.Static (tr)

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Layers.Convolution
import           Grenade.Layers.Pad
import           Grenade.Onnx
import           Grenade.Utils.ListStore

newtype PaddedConvolutionIso a = PaddedConvolutionIso {fromPaddedConvolutionIso :: a}

instance Iso PaddedConvolutionIso where
  to   = PaddedConvolutionIso
  from = fromPaddedConvolutionIso

type PaddedConvolution' (input      :: Shape)
                        (output     :: Shape)
                        (hasBias    :: HasBias)
                        (channels   :: Nat) 
                        (filters    :: Nat)
                        (kernelRows :: Nat) 
                        (kernelCols :: Nat) 
                        (strideRows :: Nat) 
                        (strideCols :: Nat) 
                        (padLeft    :: Nat) 
                        (padTop     :: Nat) 
                        (padRight   :: Nat) 
                        (padBottom  :: Nat)
  = Network
   '[ Pad padLeft padTop padRight padBottom , Convolution hasBias channels filters kernelRows kernelCols strideRows strideCols ] 
    (PaddedConvolutionShapes padLeft padTop padRight padBottom input output)

type PaddedConvolution input output hasBias channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom = Lift (PaddedConvolutionIso (PaddedConvolution' input output hasBias channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom))

type family PaddedConvolutionShapes (padLeft :: Nat) (padTop :: Nat) (padRight :: Nat) (padBottom :: Nat) (i :: Shape) (o :: Shape) :: [Shape] where
  PaddedConvolutionShapes pl pt pr pb ('D2 rows cols) output
    = '[ 'D2 rows cols , 'D2 (pt + rows + pb) (pl + cols + pr) , output ]
  PaddedConvolutionShapes pl pt pr pb ('D3 rows cols channels) output
    = '[ 'D3 rows cols channels, 'D3 (pt + rows + pb) (pl + cols + pr) channels, output ]

instance OnnxOperator (PaddedConvolutionIso (Network
   '[ Pad padLeft padTop padRight padBottom , Convolution hasBias channels filters kernelRows kernelCols strideRows strideCols ] 
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
   '[ Pad padLeft padTop padRight padBottom , Convolution 'WithoutBias channels filters kernelRows kernelCols strideRows strideCols ] 
   '[ 'D3 inputRows inputCols channels, 'D3 convInputRows convInputCols channels, 'D3 outputRows outputCols filters ])) where

  loadOnnxNode inits node = do
    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape
    
    case node `doesNotHaveAttribute` "auto_pad" of 
      Right () -> hasCorrectPadding node (Proxy :: Proxy padLeft) (Proxy :: Proxy padRight) (Proxy :: Proxy padTop) (Proxy :: Proxy padBottom)
      Left _   -> pure () -- todo: check the value of the attribute

    case node ^. #input of
      [_, w] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        return $ PaddedConvolutionIso (Pad :~> Convolution filterWeights mkListStore :~> NNil)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]


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
   '[ Pad padLeft padTop padRight padBottom , Convolution 'WithBias channels filters kernelRows kernelCols strideRows strideCols ] 
   '[ 'D3 inputRows inputCols channels, 'D3 convInputRows convInputCols channels, 'D3 outputRows outputCols filters ])) where

  loadOnnxNode inits node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape
    hasCorrectPadding node (Proxy :: Proxy padLeft) (Proxy :: Proxy padRight) (Proxy :: Proxy padTop) (Proxy :: Proxy padBottom)

    case node ^. #input of
      [_, w, b] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        filterBiases  <- readInitializerVector inits b
        return $ PaddedConvolutionIso (Pad :~> BiasConvolution filterWeights filterBiases mkListStore :~> NNil)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]
