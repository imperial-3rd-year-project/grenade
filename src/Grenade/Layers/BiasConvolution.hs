{-# LANGUAGE CPP                   #-}
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
{-# LANGUAGE OverloadedLabels      #-}
{-# LANGUAGE OverloadedStrings     #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Grenade.Layers.BiasConvolution where

import           Data.Function                       ((&))
import           Data.Kind                           (Type)
import           Data.List                           (foldl1')
import           Data.Maybe
import           Data.Proxy
import           GHC.TypeLits
import           Numeric.LinearAlgebra               hiding (konst, uniformSample, R)
import qualified Numeric.LinearAlgebra               as LA
import           Numeric.LinearAlgebra.Static        hiding ((&), build, toRows, (|||))
import           Lens.Micro                          ((^.))

import           Grenade.Core
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Update
import           Grenade.Onnx
import           Grenade.Utils.ListStore

data BiasConvolution :: Bool
                     -> Nat -- Number of channels, for the first layer this could be RGB for instance.
                     -> Nat -- Number of filters, this is the number of channels output by the layer.
                     -> Nat -- The number of rows in the kernel filter
                     -> Nat -- The number of column in the kernel filter
                     -> Nat -- The row stride of the convolution filter
                     -> Nat -- The columns stride of the convolution filter
                     -> Type where
  NoBiasConvolution :: ( KnownNat channels
                       , KnownNat filters
                       , KnownNat kernelRows
                       , KnownNat kernelColumns
                       , KnownNat strideRows
                       , KnownNat strideColumns
                       , KnownNat kernelFlattened
                       , kernelFlattened ~ (kernelRows * kernelColumns * channels))
                    => !(L kernelFlattened filters) -- The kernel filter weights
                    -> !(ListStore (L kernelFlattened filters)) -- The last kernel update (or momentum)
                    -> BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns

  BiasConvolution :: ( KnownNat channels
                     , KnownNat filters
                     , KnownNat kernelRows
                     , KnownNat kernelColumns
                     , KnownNat strideRows
                     , KnownNat strideColumns
                     , KnownNat kernelFlattened
                     , kernelFlattened ~ (kernelRows * kernelColumns * channels))
                  => !(L kernelFlattened filters) -- The kernel filter weights
                  -> !(R filters) -- The bias weights
                  -> !(ListStore (L kernelFlattened filters)) -- The last kernel update (or momentum)
                  -> BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns

data BiasConvolution' :: Bool
                      -> Nat -- Number of channels, for the first layer this could be RGB for instance.
                      -> Nat -- Number of filters, this is the number of channels output by the layer.
                      -> Nat -- The number of rows in the kernel filter
                      -> Nat -- The number of column in the kernel filter
                      -> Nat -- The row stride of the convolution filter
                      -> Nat -- The columns stride of the convolution filter
                      -> Type where
  NoBiasConvolution' :: ( KnownNat channels
                        , KnownNat filters
                        , KnownNat kernelRows
                        , KnownNat kernelColumns
                        , KnownNat strideRows
                        , KnownNat strideColumns
                        , KnownNat kernelFlattened
                        , kernelFlattened ~ (kernelRows * kernelColumns * channels))
                     => !(L kernelFlattened filters) -- The kernel filter weights
                     -> BiasConvolution' 'False channels filters kernelRows kernelColumns strideRows strideColumns

  BiasConvolution' :: ( KnownNat channels
                      , KnownNat filters
                      , KnownNat kernelRows
                      , KnownNat kernelColumns
                      , KnownNat strideRows
                      , KnownNat strideColumns
                      , KnownNat kernelFlattened
                      , kernelFlattened ~ (kernelRows * kernelColumns * channels))
                   => !(L kernelFlattened filters) -- The kernel filter weights
                   -> R filters -- The bias weights
                   -> BiasConvolution' 'True channels filters kernelRows kernelColumns strideRows strideColumns

{---------------------------}
{--    Layer instances    --}
{---------------------------}

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         , kernelFlattened ~ (kernelRows * kernelColumns * channels)
         ) => UpdateLayer (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) = (BiasConvolution' 'False channels filters kernelRows kernelColumns strideRows strideColumns)

  type MomentumStore (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns)  = ListStore (L (kernelRows * kernelColumns * channels) filters)

  runUpdate opt@OptSGD{} x@(NoBiasConvolution oldKernel store) (NoBiasConvolution' kernelGradient) =
    let  momentum = getData opt x store
         result = descendMatrix opt (MatrixValuesSGD oldKernel kernelGradient momentum)
         newStore = setData opt x store (matrixMomentum result)
    in NoBiasConvolution (matrixActivations result) newStore
  runUpdate opt@OptAdam{} x@(NoBiasConvolution oldKernel store) (NoBiasConvolution' kernelGradient) =
    let (m, v) = toTuple $ getData opt x store
        result = descendMatrix opt (MatrixValuesAdam (getStep store) oldKernel kernelGradient m v)
        newStore = setData opt x store [matrixM result, matrixV result]
    in NoBiasConvolution (matrixActivations result) newStore
    where toTuple [m ,v] = (m, v)
          toTuple xs = error $ "unexpected input of length " ++ show (length xs) ++ "in toTuple in Convolution.hs"

  reduceGradient grads = NoBiasConvolution' $ dmmap (/ (fromIntegral $ length grads)) (foldl1' add (map (\(NoBiasConvolution' x) -> x) grads))

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         , kernelFlattened ~ (kernelRows * kernelColumns * channels)
         ) => UpdateLayer (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) = (BiasConvolution' 'False channels filters kernelRows kernelColumns strideRows strideColumns)

  type MomentumStore (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns)  = ListStore (L (kernelRows * kernelColumns * channels) filters)

  runUpdate      = undefined
  reduceGradient = undefined

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
--   Case without bias vector.
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
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * filters)
         ) => Layer (BiasConvolution 'False channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (BiasConvolution 'False channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (NoBiasConvolution kernel _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col kx ky sx sy ix iy ex
        mt = c LA.<> ek
        r  = col2vid 1 1 1 1 ox oy mt
        rs = fromJust . create $ r
    in  (S3D input, S3D rs)
  runBackwards (NoBiasConvolution kernel _) (S3D input) (S3D dEdy) =
    let ex = extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col kx ky sx sy ix iy ex

        eo = extract dEdy
        ek = extract kernel

        vs = vid2col 1 1 1 1 ox oy eo

        kN = fromJust . create $ tr c LA.<> vs

        dW = vs LA.<> tr ek

        xW = col2vid kx ky sx sy ix iy dW
    in  (NoBiasConvolution' kN, S3D . fromJust . create $ xW)


-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
--   Case with bias vector
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
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * filters)
         ) => Layer (BiasConvolution 'True channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (BiasConvolution 'True channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (BiasConvolution kernel biases _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)

        c  = vid2col kx ky sx sy ix iy ex
        mt = c LA.<> ek
        bs = LA.build (ox * fs, oy) (\i _ -> extract biases ! div (round i) ox)
        r  = col2vid 1 1 1 1 ox oy mt
        rs = fromJust . create $ (r + bs)
    in  (S3D input, S3D rs)
  runBackwards = error "Backpropagation not implemented for convolutional layers with bias."

-- | A two dimentional image may have a convolution filter applied to it.
--   Case without bias vector
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * filters)
         ) => Layer (BiasConvolution 'False 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (BiasConvolution 'False 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a convolution filter applied to it.
--   Case with bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * filters)
         ) => Layer (BiasConvolution 'True 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (BiasConvolution 'True 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimensional image may have a convolution filter applied to it producing
--   a two dimensional image if both channels and filters is 1.
--   Case without bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         ) => Layer (BiasConvolution 'False 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (BiasConvolution 'False 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

-- | A two dimensional image may have a convolution filter applied to it producing
--   a two dimensional image if both channels and filters is 1.
--   Case with bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         ) => Layer (BiasConvolution 'True 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (BiasConvolution 'True 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

-- | A three dimensional image can produce a 2D image from a convolution with 1 filter
--   Case without bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , KnownNat channels
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * 1)
         ) => Layer (BiasConvolution 'False channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (BiasConvolution 'False channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

-- | A three dimensional image can produce a 2D image from a convolution with 1 filter
--   Case with bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , KnownNat channels
         , strideRows * (outputRows - 1) <= (inputRows - kernelRows + 1) - 1
         , (inputRows - kernelRows + 1) <= (outputRows * strideRows)
         , strideCols * (outputCols - 1) <= (inputCols - kernelCols + 1) - 1
         , (inputCols - kernelCols + 1) <= (outputCols * strideCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * 1)
         ) => Layer (BiasConvolution 'True channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (BiasConvolution 'True channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

{--------------------}
{-- ONNX Instances --}
{--------------------}

instance OnnxOperator (BiasConvolution 'True channels filters kernelRows kernelCols strideRows strideCols) where
  onnxOpTypeNames _ = ["Conv"]

instance OnnxOperator (BiasConvolution 'False channels filters kernelRows kernelCols strideRows strideCols) where
  onnxOpTypeNames _ = ["Conv"]

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat channels
         , KnownNat (kernelRows * kernelCols * channels)
         ) => OnnxLoadable (BiasConvolution 'True channels filters kernelRows kernelCols strideRows strideCols) where
  loadOnnxNode inits node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape
    (node & hasCorrectPadding) (Proxy :: Proxy 0) (Proxy :: Proxy 0) (Proxy :: Proxy 0) (Proxy :: Proxy 0)

    case node ^. #input of
      [_, w, b] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        filterBias    <- readInitializerVector inits b
        return (BiasConvolution filterWeights filterBias mkListStore)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat channels
         , KnownNat (kernelRows * kernelCols * channels)
         ) => OnnxLoadable (BiasConvolution 'False channels filters kernelRows kernelCols strideRows strideCols) where
  loadOnnxNode inits node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape
    hasCorrectPadding node (Proxy :: Proxy 0) (Proxy :: Proxy 0) (Proxy :: Proxy 0) (Proxy :: Proxy 0)

    case node ^. #input of
      [_, w] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        return (NoBiasConvolution filterWeights mkListStore)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]

{-------------------------}
{-- Optimiser Instances --}
{-------------------------}

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) =>
         LayerOptimizerData (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) where
  type MomentumExpOptResult (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  type MomentumDataType (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = konst 0

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) =>
         LayerOptimizerData (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) where
  type MomentumExpOptResult (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  type MomentumDataType (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  getData = undefined
  setData = undefined
  newData = undefined

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => LayerOptimizerData (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) where
  type MomentumExpOptResult (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam)  = [L (kernelRows * kernelColumns * channels) filters]
  type MomentumDataType (BiasConvolution 'False channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) = L (kernelRows * kernelColumns * channels) filters
  getData = getListStore
  setData = setListStore
  newData _ _ = konst 0

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => LayerOptimizerData (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) where
  type MomentumExpOptResult (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam)  = [L (kernelRows * kernelColumns * channels) filters]
  type MomentumDataType (BiasConvolution 'True channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) = L (kernelRows * kernelColumns * channels) filters
  getData = undefined
  setData = undefined
  newData = undefined
