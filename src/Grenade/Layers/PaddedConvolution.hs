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

import           Control.Monad           (guard)
import           Data.Proxy
import qualified Data.Text               as T
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static (tr)

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Layers.Convolution
import           Grenade.Layers.Pad
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.Graph
import           Grenade.Utils.ListStore

import           Grenade.Onnx.Iso
import           Lens.Micro ((^.))
import Debug.Trace
import qualified Proto.Onnx as P

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

   -- TODO: Check how auto_pad, group and optional bias should be supported.
   --       I don't think we can support group without reshape layers probably.
   --
   --   size of w is filters x channels x kernelRows x kernelCols
  loadOnnxNode inits node = do
    doesNotHaveAttribute  node "auto_pad"
    -- the below line doesnt work because the default value for `group` is 1
    -- doesNotHaveAttribute  node "group"
    hasSupportedDilations node
    hasMatchingShape  node "kernel_shape" kernelShape
    hasMatchingShape  node "strides"     strideShape

    hasCorrectPadding node (Proxy :: Proxy padLeft) (Proxy :: Proxy padRight) (Proxy :: Proxy padTop) (Proxy :: Proxy padBottom)

    (_ : w : _) <- Just (node ^. #input)

    filterWeights <- tr <$> readInitializerMatrix inits w
    return $ PaddedConvolutionIso (Pad :~> Convolution filterWeights mkListStore :~> NNil)
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]

hasSupportedDilations :: P.NodeProto -> Maybe ()
hasSupportedDilations node = case readIntsAttribute "dilations" node of
  Just ds -> guard $ all (==1) ds
  Nothing -> return ()

hasMatchingShape :: P.NodeProto -> T.Text -> [Integer] -> Maybe ()
hasMatchingShape node attribute dims = case readIntsAttribute attribute node of
                                          Just xs -> guard $ xs == map fromIntegral dims
                                          _       -> return ()

-- TODO: Add support for auto_pad
hasCorrectPadding :: (KnownNat padLeft, KnownNat padRight, KnownNat padTop, KnownNat padBottom)
                  => P.NodeProto -> Proxy padLeft -> Proxy padRight -> Proxy padTop -> Proxy padBottom -> Maybe ()
hasCorrectPadding node ppl ppr ppt ppb 
  = let left   = fromIntegral $ natVal ppl
        right  = fromIntegral $ natVal ppr
        top    = fromIntegral $ natVal ppt
        bottom = fromIntegral $ natVal ppb
     in case readIntsAttribute "pads" node of
          Just [left', top', right', bottom'] -> guard (left == left' && top == top' && right == right' && bottom == bottom')
          Nothing                             -> guard (left == 0     && top == 0    && right == 0      && bottom == 0)
          _                                   -> Nothing
