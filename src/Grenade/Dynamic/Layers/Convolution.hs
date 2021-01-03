{-# LANGUAGE CPP                   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE DataKinds             #-}
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

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Grenade.Dynamic.Layers.Convolution where

import           Control.Monad.Primitive             (PrimBase, PrimState)
import           Data.Constraint                     (Dict (..))
import           Data.Proxy
import           Data.Reflection                     (reifyNat)
import           Data.Singletons
import           Data.Singletons.Prelude.Num         ((%*))
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static        hiding (build, toRows, (&),
                                                      (|||), size)
import           System.Random.MWC                   (Gen)
import           Unsafe.Coerce                       (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic.Network
import           Grenade.Dynamic.Specification
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Convolution
import           Grenade.Utils.ListStore

{--------------------}
{-- GNum Instances --}
{--------------------}

instance (KnownNat striCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * channels)) =>
         GNum (Convolution 'WithBias padding channels filters kernelRows kernelCols strideRows striCols) where
  n |* (BiasConvolution w b store) = BiasConvolution (dmmap (fromRational n *) w) (dvmap (fromRational n *) b) (n |* store)
  (BiasConvolution w1 b1 store1)  |+ (BiasConvolution w2 b2 store2) = BiasConvolution (w1 + w2) (b1 + b2) (store1 |+ store2)
  gFromRational r = BiasConvolution (fromRational r) (fromRational r) mkListStore

instance (KnownNat striCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * channels)) =>
         GNum (Convolution 'WithoutBias padding channels filters kernelRows kernelCols strideRows striCols) where
  n |* (Convolution w store) = Convolution (dmmap (fromRational n *) w) (n |* store)
  (Convolution w1 store1)  |+ (Convolution w2 store2)  = Convolution (w1 + w2) (store1 |+ store2)
  gFromRational r = Convolution (fromRational r) mkListStore

instance (KnownNat striCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * channels)) =>
         GNum (Convolution' 'WithBias padding channels filters kernelRows kernelCols strideRows striCols) where
  n |* (BiasConvolution' g1 g2) = BiasConvolution' (dmmap (fromRational n *) g1) (dvmap (fromRational n *) g2)
  (BiasConvolution' g1 g2) |+ (BiasConvolution' g3 g4) = BiasConvolution' (g1 + g3) (g2 + g4)
  gFromRational r = BiasConvolution' (fromRational r) (fromRational r)

instance (KnownNat striCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * channels)) =>
         GNum (Convolution' 'WithoutBias padding channels filters kernelRows kernelCols strideRows striCols) where
  n |* (Convolution' g) = Convolution' (dmmap (fromRational n *) g)
  (Convolution' g) |+ (Convolution' g2) = Convolution' (g + g2)
  gFromRational r = Convolution' (fromRational r)

{------------------------------}
{-- DynamicNetwork Instances --}
{------------------------------}
instance (KnownNat channels, KnownNat filters, KnownNat kernelRows, KnownNat kernelColumns, KnownNat strideRows, KnownNat strideColumns) =>
         FromDynamicLayer (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  fromDynamicLayer inp _ _ =
    SpecNetLayer $
    SpecConvolution
      (tripleFromSomeShape inp)
      (natVal (Proxy :: Proxy channels))
      (natVal (Proxy :: Proxy filters))
      (natVal (Proxy :: Proxy kernelRows))
      (natVal (Proxy :: Proxy kernelColumns))
      (natVal (Proxy :: Proxy strideRows))
      (natVal (Proxy :: Proxy strideColumns))

instance ToDynamicLayer SpecConvolution where
  toDynamicLayer = toDynamicLayer'

toDynamicLayer' :: (PrimBase m)=> WeightInitMethod -> Gen (PrimState m) -> SpecConvolution -> m SpecNetwork
toDynamicLayer' _ _ (SpecConvolution inp@(_, 1, 1) _ _ _ _ _ _) = error $ "1D input to a deconvolutional layer is not permited! you specified: " ++ show inp
toDynamicLayer' wInit gen (SpecConvolution (rows, cols, depth) ch fil kerRows kerCols strRows strCols) =
    reifyNat ch $ \(pxCh :: (KnownNat channels) => Proxy channels) ->
    reifyNat fil $ \(pxFil :: (KnownNat filters) => Proxy filters) ->
    reifyNat kerRows $ \(pxKerRows :: (KnownNat kernelRows) => Proxy kernelRows) ->
    reifyNat kerCols $ \(pxKerCols :: (KnownNat kernelCols) => Proxy kernelCols) ->
    reifyNat strRows $ \(_ :: (KnownNat strideRows) => Proxy strideRows) ->
    reifyNat strCols $ \(_ :: (KnownNat strideCols) => Proxy strideCols) ->
    reifyNat rows $ \(pxRows :: (KnownNat rows) => Proxy rows) ->
    reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
    reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
    reifyNat ((rows - 1) * strRows + kerRows) $ \(pxOutRows :: (KnownNat outRows) => Proxy outRows) ->
    reifyNat ((cols - 1) * strCols + kerCols) $ \(_ :: (KnownNat outCols) => Proxy outCols) ->
    case ((singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxCh
         , singByProxy pxFil %* ((singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxCh)
         , singByProxy pxOutRows %* singByProxy pxFil -- 'D3 representation
         , singByProxy pxRows %* singByProxy pxCh -- 'D3 representation
         ) of
      (SNat, SNat, SNat, SNat) | ch == 1 && fil == 1 ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict (channels ~ 1, filters ~ 1, OutputShapeIsOkay rows 0 kernelRows strideRows outRows, OutputShapeIsOkay cols 0 kernelCols strideCols outCols )) of
          Dict -> do
            (layer  :: Convolution 'WithoutBias 'NoPadding 1 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat) | ch == 1 ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict (channels ~ 1, OutputShapeIsOkay rows 0 kernelRows strideRows outRows, OutputShapeIsOkay cols 0 kernelCols strideCols outCols )) of
          Dict -> do
            (layer  :: Convolution 'WithoutBias 'NoPadding 1 filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D3 outRows outCols filters))
      (SNat, SNat, SNat, SNat) | fil == 1 ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict (filters ~ 1, OutputShapeIsOkay rows 0 kernelRows strideRows outRows, OutputShapeIsOkay cols 0 kernelCols strideCols outCols )) of
          Dict -> do
            (layer  :: Convolution 'WithoutBias 'NoPadding channels 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) (sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat) ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict ( OutputShapeIsOkay rows 0 kernelRows strideRows outRows, OutputShapeIsOkay cols 0 kernelCols strideCols outCols ) ) of
          Dict -> do
            (layer  :: Convolution 'WithoutBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) (sing :: Sing ('D3 outRows outCols filters))

-- | Creates a specification for a convolutional layer with 2D input to the layer. If channels and filters are both 1 then the output is 2D otherwise it is 3D. The output sizes are `out = (in -
-- kernel) / stride + 1`, for rows and cols and the depth is filters for 3D output.
specConvolution2DInput ::
     (Integer, Integer) -- ^ Number of input rows.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the convolution filter
  -> Integer -- ^ The cols stride of the convolution filter
  -> SpecNet
specConvolution2DInput (rows, cols) = specConvolution3DInput (rows, cols, 1)

-- | Creates a specification for a convolutional layer with 3D input to the layer. If the filter is 1 then the output is 2D, otherwise it is 3D. The output sizes are `out = (in - kernel) / stride +
-- 1`, for rows and cols and the depth is filters for 3D output.
specConvolution3DInput ::
     (Integer, Integer, Integer) -- ^ Input to layer (rows, cols, depths). Use 1 if not used or the function @specConvolution1DInput@ and @specConvolution2DInput@.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the convolution filter
  -> Integer -- ^ The cols stride of the convolution filter
  -> SpecNet
specConvolution3DInput inp channels filters kernelRows kernelCols strideRows strideCols =
  SpecNetLayer $ SpecConvolution inp channels filters kernelRows kernelCols strideRows strideCols

-- | A convolution layer. 2D and 3D input/output only!
convolution ::
     Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the convolution filter
  -> Integer -- ^ The cols stride of the convolution filter
  -> BuildM ()
convolution channels filters kernelRows kernelCols strideRows strideCols = do
  inp@(r, c, _) <- buildRequireLastLayerOut IsNot1D
  let outRows = (r - kernelRows) `div` strideRows + 1
      outCols = (c - kernelCols) `div` strideCols + 1
  buildAddSpec $ SpecNetLayer $ SpecConvolution inp channels filters kernelRows kernelCols strideRows strideCols
  buildSetLastLayer (outRows, outCols, filters)
