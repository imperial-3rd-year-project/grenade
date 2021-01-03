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

module Grenade.Dynamic.Layers.Deconvolution 
  ( SpecDeconvolution (..)
  , specDeconvolution2DInput
  , specDeconvolution3DInput
  , deconvolution
  ) where

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
import           Grenade.Dynamic.Internal.Build
import           Grenade.Dynamic.Specification
import           Grenade.Dynamic.Network
import           Grenade.Layers.Deconvolution
import           Grenade.Utils.ListStore

-------------------- DynamicNetwork instance --------------------

instance (KnownNat channels, KnownNat filters, KnownNat kernelRows, KnownNat kernelColumns, KnownNat strideRows, KnownNat strideColumns) =>
         FromDynamicLayer (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  fromDynamicLayer inp _ _ =
    SpecNetLayer $
    SpecDeconvolution
      (tripleFromSomeShape inp)
      (natVal (Proxy :: Proxy channels))
      (natVal (Proxy :: Proxy filters))
      (natVal (Proxy :: Proxy kernelRows))
      (natVal (Proxy :: Proxy kernelColumns))
      (natVal (Proxy :: Proxy strideRows))
      (natVal (Proxy :: Proxy strideColumns))


instance ToDynamicLayer SpecDeconvolution where
  toDynamicLayer  = toDynamicLayer'

toDynamicLayer' :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> SpecDeconvolution -> m SpecNetwork
toDynamicLayer' _ _ (SpecDeconvolution inp@(_, 1, 1) _ _ _ _ _ _) = error $ "1D input to a deconvolutional layer is not permited! you specified: " ++ show inp
toDynamicLayer' wInit gen (SpecDeconvolution (rows, cols, depth) ch fil kerRows kerCols strRows strCols) =
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
    case ( (singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxFil
         , (singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxCh -- this is the input: i = (kernelRows * kernelCols) * channels)
         , singByProxy pxCh %* ((singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxFil)
         , singByProxy pxOutRows %* singByProxy pxFil -- 'D3 representation
         , singByProxy pxRows %* singByProxy pxCh -- 'D3 representation
         ) of
      (SNat, SNat, SNat, SNat, SNat) | ch == 1 && fil == 1 && depth == 0 ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (channels ~ 1, filters ~ 1, ((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer  :: Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat, SNat) | ch == 1 ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (channels ~ 1, ((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer  :: Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D3 outRows outCols filters))
      (SNat, SNat, SNat, SNat, SNat) | fil == 1 ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (filters ~ 1, ((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer  :: Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) ( sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat, SNat) ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer :: Deconvolution channels filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) ( sing :: Sing ('D3 outRows outCols filters))


-- | Creates a specification for a deconvolutional layer with 2D input to the layer. If channels and filters are both 1 then the output is 2D otherwise it is 3D. The output sizes are `out = (in - 1) *
-- stride + kernel`, for rows and cols and the depth is filters for 3D output.
specDeconvolution2DInput ::
     (Integer, Integer) -- ^ Number of input rows.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the deconvolution filter
  -> Integer -- ^ The cols stride of the deconvolution filter
  -> SpecNet
specDeconvolution2DInput (rows, cols) = specDeconvolution3DInput (rows, cols, 1)

-- | Creates a specification for a deconvolutional layer with 3D input to the layer. If the filter is 1 then the output is 2D, otherwise it is 3D. The output sizes are `out = (in - 1) * stride +
-- kernel`, for rows and cols and the depth is filters for 3D output.
specDeconvolution3DInput ::
     (Integer, Integer, Integer) -- ^ Input to layer (rows, cols, depths). Use 1 if not used or the function @specDeconvolution1DInput@ and @specDeconvolution2DInput@.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the deconvolution filter
  -> Integer -- ^ The cols stride of the deconvolution filter
  -> SpecNet
specDeconvolution3DInput inp channels filters kernelRows kernelCols strideRows strideCols =
  SpecNetLayer $ SpecDeconvolution inp channels filters kernelRows kernelCols strideRows strideCols


-- | A deconvolution layer. 2D and 3D input/output only!
deconvolution ::
     Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the deconvolution filter
  -> Integer -- ^ The cols stride of the deconvolution filter
  -> BuildM ()
deconvolution channels filters kernelRows kernelCols strideRows strideCols = do
  inp@(r, c, _) <- buildRequireLastLayerOut IsNot1D
  let outRows = (r - 1) * strideRows + kernelRows
      outCols = (c - 1) * strideCols + kernelCols
  buildAddSpec $ SpecNetLayer $ SpecDeconvolution inp channels filters kernelRows kernelCols strideRows strideCols
  buildSetLastLayer (outRows, outCols, filters)

-------------------- GNum instances --------------------


instance (KnownNat strideCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * filters)) =>
         GNum (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) where
  n |* (Deconvolution w store) = Deconvolution (dmmap (fromRational n *) w) (n |* store)
  (Deconvolution w1 store1) |+ (Deconvolution w2 store2) = Deconvolution (w1 + w2) (store1 |+ store2)
  gFromRational r = Deconvolution (fromRational r) mkListStore


instance (KnownNat strideCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * filters)) =>
         GNum (Deconvolution' channels filters kernelRows kernelCols strideRows strideCols) where
  n |* (Deconvolution' g) = Deconvolution' (dmmap (fromRational n *) g)
  (Deconvolution' g) |+ (Deconvolution' g2) = Deconvolution' (g + g2)
  gFromRational r = Deconvolution' (fromRational r)
