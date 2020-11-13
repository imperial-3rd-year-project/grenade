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

module Grenade.Layers.PaddedConvolution (
    PaddedConvolution (..)
  ) where

import           Control.Monad           (guard)
import           Data.Proxy
import           Data.Singletons.Prelude (Head, Last)
import qualified Data.Text               as T
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static (tr)

import           Grenade.Core.Layer
import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Layers.Convolution
import           Grenade.Layers.Pad
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.Graph
import           Grenade.Utils.ListStore

import qualified Proto.Onnx as P

--newtype PaddedConvolution' a = PaddedConvolution' {fromPaddedConvolution' :: a}
--
--instance Iso PaddedConvolution' where
--  to   = PaddedConvolution'
--  from = fromPaddedConvolution'
--
--type PaddedConvolution (input      :: Shape)
--                       (output     :: Shape)
--                       (channels   :: Nat) 
--                       (filters    :: Nat)
--                       (kernelRows :: Nat) 
--                       (kernelCols :: Nat) 
--                       (strideRows  :: Nat) 
--                       (strideCols  :: Nat) 
--                       (padLeft    :: Nat) 
--                       (padTop     :: Nat) 
--                       (padRight   :: Nat) 
--                       (padBottom  :: Nat)
--  = Network
--   '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] 
--    (PaddedConvolutionShapes padLeft padTop padRight padBottom input output)
--
--type family PaddedConvolutionShapes (padLeft :: Nat) (padTop :: Nat) (padRight :: Nat) (padBottom :: Nat) (i :: Shape) (o :: Shape) :: [Shape] where
--  PaddedConvolutionShapes pl pt pr pb ('D2 rows cols) output
--    = '[ 'D2 rows cols , 'D2 (pt + rows + pb) (pl + cols + pr) , output ]
--  PaddedConvolutionShapes pl pt pr pb ('D3 rows cols channels) output
--    = '[ 'D3 rows cols channels, 'D3 (pt + rows + pb) (pl + cols + pr) channels, output ]
--
--
--instance ( KnownNat kernelRows
--         , KnownNat kernelCols
--         , KnownNat filters
--         , KnownNat strideRows
--         , KnownNat strideCols
--         , KnownNat inputRows
--         , KnownNat inputCols
--         , KnownNat outputRows
--         , KnownNat outputCols
--         , KnownNat channels
--         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
--         , ((outputCols - 1) * rolStride) ~ (inputCols - kernelCols)
--         , KnownNat (kernelRows * kernelCols * channels)
--         , KnownNat (outputRows * filters)
--         , KnownNat padLeft
--         , KnownNat padTop
--         , KnownNat padRight
--         , KnownNat padBottom
--         ) => OnnxLoadable (PaddedConvolution input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom) where



{-----------------------------------------------------------------------}
{---           DON'T LOOK BELOW THIS, IT WAS ALL A MISTAKE           ---}
{--- ARGUABLY THE STUFF ABOVE WAS ALSO A MISTAKE, BUT LESS SO I FEEL ---}
{-----------------------------------------------------------------------}

newtype PaddedConvolution (input      :: Shape)
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
  = PaddedConvolution (Network
  '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] 
   (PaddedConvolutionShapes padLeft padTop padRight padBottom input output))

type family PaddedConvolutionShapes (padLeft :: Nat) (padTop :: Nat) (padRight :: Nat) (padBottom :: Nat) (i :: Shape) (o :: Shape) :: [Shape] where
  PaddedConvolutionShapes pl pt pr pb ('D2 rows cols) output
    = '[ 'D2 rows cols , 'D2 (pt + rows + pb) (pl + cols + pr) , output ]
  PaddedConvolutionShapes pl pt pr pb ('D3 rows cols channels) output
    = '[ 'D3 rows cols channels, 'D3 (rows + pt + pb) (cols + pl + pr) channels, output ]

instance UpdateLayer (PaddedConvolution input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom) where
  type Gradient (PaddedConvolution input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom)
    = Gradient (Network '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] (PaddedConvolutionShapes padLeft padTop padRight padBottom input output))

  runUpdate opt (PaddedConvolution net) grad = PaddedConvolution (runUpdate opt net grad)
  runSettingsUpdate settings (PaddedConvolution net) = PaddedConvolution (runSettingsUpdate settings net)
  reduceGradient = reduceGradient @(Network '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] (PaddedConvolutionShapes padLeft padTop padRight padBottom input output))

instance (i ~ Head (PaddedConvolutionShapes padLeft padTop padRight padBottom input output), o ~ Last (PaddedConvolutionShapes padLeft padTop padRight padBottom input output))
  => Layer (PaddedConvolution input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom) i o where
  type Tape (PaddedConvolution input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom) i o = Tape (Network '[ Pad padLeft padTop padRight padBottom , Convolution channels filters kernelRows kernelCols strideRows strideCols ] (PaddedConvolutionShapes padLeft padTop padRight padBottom input output)) i o
   
  runForwards (PaddedConvolution net) = runForwards net
  runBackwards (PaddedConvolution net) = runBackwards net

instance OnnxOperator (PaddedConvolution input output channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom) where
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
         , ((outputRows - 1) * strideRows) ~ (convInputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (convInputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * filters)
         , KnownNat padLeft
         , KnownNat padRight
         , KnownNat padTop
         , KnownNat padBottom
         ) => OnnxLoadable (PaddedConvolution ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) channels filters kernelRows kernelCols strideRows strideCols padLeft padTop padRight padBottom) where
  -- TODO: Check how auto_pad, group and optional bias should be supported.
  --       I don't think we can support group without reshape layers probably.
  --
--   size of w is filters x channels x kernelRows x kernelCols
  loadOnnxNode inits node = do
    doesNotHaveAttribute  node "autoPad"
    doesNotHaveAttribute  node "group"
    hasSupportedDilations node
    hasMatchingShape  node "kernelShape" kernelShape
    hasMatchingShape  node "strides"     strideShape
    hasCorrectPadding node (Proxy :: Proxy padLeft) (Proxy :: Proxy padRight) (Proxy :: Proxy padTop) (Proxy :: Proxy padBottom)

    filterWeights <- tr <$> readInitializerMatrix inits "W"
    return $ PaddedConvolution (Pad :~> Convolution filterWeights mkListStore :~> NNil)
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
