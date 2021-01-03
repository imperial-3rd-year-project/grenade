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
{-|
Module      : Grenade.Layers.Deconvolution
Description : Deconvolution layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

A deconvolution layer is in many ways a convolution layer in reverse.
It learns a kernel to apply to each pixel location, spreading it out
into a larger layer.

This layer is important for image generation tasks, such as GANs on
images.
-}
module Grenade.Layers.Deconvolution (
    Deconvolution (..)
  , Deconvolution' (..)
  ) where

import           Control.DeepSeq                     (NFData (..))
import           Data.Kind                           (Type)
import           Data.List                           (foldl1')
import           Data.Maybe
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits

import           Numeric.LinearAlgebra               hiding (konst,
                                                      uniformSample)
import qualified Numeric.LinearAlgebra               as LA
import           Numeric.LinearAlgebra.Static        hiding (build, toRows,
                                                      (|||))

import           Grenade.Core
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Update
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

-- | A Deconvolution layer for a neural network.
--   This uses the im2col Convolution trick popularised by Caffe.
--
--   The Deconvolution layer is a way of spreading out a single response
--   into a larger image, and is useful in generating images.
--
data Deconvolution :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                   -> Nat -- Number of filters, this is the number of channels output by the layer.
                   -> Nat -- The number of rows in the kernel filter
                   -> Nat -- The number of column in the kernel filter
                   -> Nat -- The row stride of the Deconvolution filter
                   -> Nat -- The columns stride of the Deconvolution filter
                   -> Type where
  Deconvolution :: ( KnownNat channels
                   , KnownNat filters
                   , KnownNat kernelRows
                   , KnownNat kernelColumns
                   , KnownNat strideRows
                   , KnownNat strideColumns
                   , KnownNat kernelFlattened
                   , kernelFlattened ~ (kernelRows * kernelColumns * filters))
                 => !(L kernelFlattened channels) -- The kernel filter weights
                 -> !(ListStore (L kernelFlattened channels)) -- The last kernel update (or momentum)
                 -> Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  rnf (Deconvolution a b) = rnf a `seq` rnf b

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => Serialize (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Deconvolution w store) = do
    putListOf put . toList . flatten . extract $ w
    put (fmap (toList . flatten . extract) store)
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy channels)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      store <- fmap (fromMaybe (error "Vector of incorrect size") . create . reshape f . LA.fromList)  <$> get
      return $ Deconvolution wN store


data Deconvolution' :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                    -> Nat -- Number of filters, this is the number of channels output by the layer.
                    -> Nat -- The number of rows in the kernel filter
                    -> Nat -- The number of column in the kernel filter
                    -> Nat -- The row stride of the Deconvolution filter
                    -> Nat -- The columns stride of the Deconvolution filter
                    -> Type where
  Deconvolution' :: ( KnownNat channels
                    , KnownNat filters
                    , KnownNat kernelRows
                    , KnownNat kernelColumns
                    , KnownNat strideRows
                    , KnownNat strideColumns
                    , KnownNat kernelFlattened
                    , kernelFlattened ~ (kernelRows * kernelColumns * filters))
                 => !(L kernelFlattened channels) -- The kernel filter gradient
                 -> Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns) where
  rnf (Deconvolution' a) = rnf a `seq` ()

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) =>
         Serialize (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Deconvolution' w) = putListOf put . toList . flatten . extract $ w
  get = do
    let f = fromIntegral $ natVal (Proxy :: Proxy channels)
    wN <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
    return $ Deconvolution' wN

instance (KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns) => FoldableGradient (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns) where
  mapGradient f (Deconvolution' kernelGradient) = Deconvolution' (dmmap f kernelGradient)
  squaredSums (Deconvolution' kernelGradient) = [sumM . squareM $ kernelGradient]


instance Show (Deconvolution c f k k' s s') where
  show (Deconvolution a _) = renderConv a
    where
      renderConv mm =
        let m = extract mm
            ky = fromIntegral $ natVal (Proxy :: Proxy k)
            rs = LA.toColumns m
            ms = map (take ky) $ toLists . reshape ky <$> rs
            render n'
              | n' <= 0.2 = ' '
              | n' <= 0.4 = '.'
              | n' <= 0.6 = '-'
              | n' <= 0.8 = '='
              | otherwise = '#'
            px = (fmap . fmap . fmap) render ms
         in unlines $ foldl1 (zipWith (\a' b' -> a' ++ "   |   " ++ b')) px

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat ((kernelRows * kernelColumns) * filters)
         , KnownNat ((kernelRows * kernelColumns) * channels)
         , KnownNat (channels * ((kernelRows * kernelColumns) * filters))
         ) =>
         RandomLayer (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  createRandomWith m gen = do
    wN <- getRandomMatrix i i m gen
    return $ Deconvolution wN mkListStore
    where
      i = natVal (Proxy :: Proxy ((kernelRows * kernelColumns) * channels))


instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => UpdateLayer (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) = (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns)

  type MomentumStore (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns)  = ListStore (L (kernelRows * kernelColumns * filters) channels)

  runUpdate opt@OptSGD{} x@(Deconvolution oldKernel store) (Deconvolution' kernelGradient) =
    let oldMomentum = getData opt x store
        result = descendMatrix opt (MatrixValuesSGD oldKernel kernelGradient oldMomentum)
        newStore = setData opt x store (matrixMomentum result)
    in Deconvolution (matrixActivations result) newStore
  runUpdate opt@OptAdam{} x@(Deconvolution oldKernel store) (Deconvolution' kernelGradient) =
    let (m, v) = toTuple $ getData opt x store
        result = descendMatrix opt (MatrixValuesAdam (getStep store) oldKernel kernelGradient m v)
        newStore = setData opt x store [matrixM result, matrixV result]
    in Deconvolution (matrixActivations result) newStore
    where toTuple [m ,v] = (m, v)
          toTuple xs = error $ "unexpected input of length " ++ show (length xs) ++ "in toTuple in Convolution.hs"

  reduceGradient grads = Deconvolution' $ dmmap (/ (fromIntegral $ length grads)) (foldl1' add (map (\(Deconvolution' x) -> x) grads))



instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) =>
         LayerOptimizerData (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) where
  type MomentumExpOptResult (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * filters) channels
  type MomentumDataType (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * filters) channels
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = konst 0

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => LayerOptimizerData (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) where
  type MomentumExpOptResult (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam)  = [L (kernelRows * kernelColumns * filters) channels]
  type MomentumDataType (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) = L (kernelRows * kernelColumns * filters) channels
  getData = getListStore
  setData = setListStore
  newData _ _ = konst 0


-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * filters)
         , KnownNat (outputRows * filters)
         ) => Layer (Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         ) => Layer (Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D fore :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D fore)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         , KnownNat channels
         ) => Layer (Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D fore :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D fore)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized Deconvolution filter run across it.
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
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * filters)
         , KnownNat (outputRows * filters)
         ) => Layer (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Deconvolution kernel _) (S3D input) =
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

        c  = vid2col 1 1 1 1 ix iy ex

        mt = c LA.<> tr ek

        r  = col2vid kx ky sx sy ox oy mt
        rs = fromJust . create $ r
    in  (S3D input, S3D rs)
  runBackwards (Deconvolution kernel _) (S3D input) (S3D dEdy) =
    let ex = extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col 1 1 1 1 ix iy ex

        eo = extract dEdy
        ek = extract kernel

        vs = vid2col kx ky sx sy ox oy eo

        kN = fromJust . create . tr $ tr c LA.<> vs

        dW = vs LA.<> ek

        xW = col2vid 1 1 1 1 ix iy dW
    in  (Deconvolution' kN, S3D . fromJust . create $ xW)
