{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# HLINT ignore "Use camelCase"      #-}
{-|
Module      : Grenade.Layers.Internal.Add
Description : Fast functions for performing convolutions and helper functions
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Internal.Convolution (
  -- * im2col functions
    im2col
  , col2im
  , col2vid
  , vid2col

  -- * convolution functions
  , forwardConv2d
  , forwardBiasConv2d
  , backwardConv2d
  , backwardBiasConv2d
  ) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray,
                                              withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Grenade.Types
import           Numeric.LinearAlgebra       (Matrix, Vector, cols, flatten,
                                              rows)
import qualified Numeric.LinearAlgebra       as LA
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import Grenade.Layers.Internal.Hmatrix

-- | Rearrange the matrix columns into blocks, calculates the number of channels automatically.
col2vid :: Int            -- ^ kernel rows
        -> Int            -- ^ kernel columns 
        -> Int            -- ^ stride rows
        -> Int            -- ^ stride columns
        -> Int            -- ^ input height
        -> Int            -- ^ input width
        -> Matrix RealNum -- ^ input matrix
        -> Matrix RealNum -- ^ output matrix 
col2vid kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = cols dataCol `div` (kernelRows * kernelColumns)
      outRows  = (height - kernelRows) `div` strideRows + 1
      outCols  = (width - kernelColumns) `div` strideColumns + 1
  in  col2im_c channels height width kernelRows kernelColumns strideRows strideColumns 0 0 outRows outCols dataCol

-- | Rearrange the matrix columns into blocks, assumes there is only one channel
col2im :: Int            -- ^ kernel rows
       -> Int            -- ^ kernel columns 
       -> Int            -- ^ stride rows
       -> Int            -- ^ stride columns
       -> Int            -- ^ input height
       -> Int            -- ^ input width
       -> Matrix RealNum -- ^ input matrix
       -> Matrix RealNum -- ^ output matrix 
col2im kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = 1
      outRows  = (height - kernelRows) `div` strideRows + 1
      outCols  = (width - kernelColumns) `div` strideColumns + 1
  in  col2im_c channels height width kernelRows kernelColumns strideRows strideColumns 0 0 outRows outCols dataCol

col2im_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
col2im_c channels height width kernelRows kernelColumns strideRows strideColumns padl padt outRows outCols dataCol =
  let vec = flatten dataCol
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (height * width * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        col2im_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns padt padl outRows outCols outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (height * width * channels)
    return $ U.matrixFromVector U.RowMajor (height * channels) width matVec
{-# INLINE col2im_c #-}

foreign import ccall unsafe
    col2im_cpu
      :: Ptr RealNum -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()

-- | Rearrange the matrix blocks into columns, calculates the number of channels automatically.
vid2col :: Int            -- ^ kernel rows
        -> Int            -- ^ kernel columns 
        -> Int            -- ^ stride rows
        -> Int            -- ^ stride columns
        -> Int            -- ^ input height
        -> Int            -- ^ input width
        -> Matrix RealNum -- ^ input matrix
        -> Matrix RealNum -- ^ output matrix 
vid2col kernelRows kernelColumns strideRows strideColumns height width dataVid =
  let channels = rows dataVid `div` height
      rowOut          = (height - kernelRows) `div` strideRows + 1
      colOut          = (width - kernelColumns) `div` strideColumns + 1
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns 0 0 dataVid rowOut colOut

-- | Rearrange the matrix blocks into columns, assumes there is only one channel
im2col :: Int            -- ^ kernel rows
       -> Int            -- ^ kernel columns 
       -> Int            -- ^ stride rows
       -> Int            -- ^ stride columns
       -> Matrix RealNum -- ^ input matrix
       -> Matrix RealNum -- ^ output matrix 
im2col kernelRows kernelColumns strideRows strideColumns dataIm =
  let channels = 1
      height = rows dataIm
      width  = cols dataIm
      rowOut          = (height - kernelRows) `div` strideRows + 1
      colOut          = (width - kernelColumns) `div` strideColumns + 1
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns 0 0 dataIm rowOut colOut

im2col_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Int -> Int -> Matrix RealNum
im2col_c channels height width kernelRows kernelColumns strideRows strideColumns padl padt dataIm outRows outCols =
  let vec             = flatten dataIm
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = outRows * outCols
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (numberOfPatches * kernelSize * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        im2col_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns padt padl outRows outCols outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (numberOfPatches * kernelSize * channels)
    return $ U.matrixFromVector U.RowMajor (kernelSize * channels) numberOfPatches matVec
{-# INLINE im2col_c #-}

foreign import ccall unsafe
    im2col_cpu
      :: Ptr RealNum -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()

foreign import ccall unsafe
    in_place_add_per_channel_cpu
      :: Ptr RealNum -> Int -> Int -> Int -> Ptr RealNum -> IO ()

-- | Efficient implementation of convolutions using the im2col trick as popularised by Caffe
forwardConv2d :: Matrix RealNum -- ^ input
              -> Int            -- ^ input channels
              -> Int            -- ^ input rows
              -> Int            -- ^ input columns
              -> Matrix RealNum -- ^ kernel
              -> Int            -- ^ kernel filters
              -> Int            -- ^ kernel rows
              -> Int            -- ^ kernel columns
              -> Int            -- ^ stride height
              -> Int            -- ^ stride width
              -> Int            -- ^ output rows
              -> Int            -- ^ output columns
              -> Int            -- ^ pad left
              -> Int            -- ^ pad top
              -> Matrix RealNum -- ^ output of convolution
forwardConv2d input channels rows cols kernel filters kernelRows kernelCols strideRows strideCols outRows outCols padLeft padTop
  = let dataCol  = im2col_c channels rows cols kernelRows kernelCols strideRows strideCols padLeft padTop input outRows outCols
        gemmM    = LA.tr kernel LA.<> dataCol
        dataVid  = reshapeMatrix (filters * outRows) outCols gemmM
    in  dataVid

-- | Efficient implementation of convolutions using the im2col trick as popularised by Caffe
--   also adds some bias to the functions, note that it does it in place to prevent unnecessary allocation.
forwardBiasConv2d :: Matrix RealNum -- ^ input
                  -> Int            -- ^ input channels
                  -> Int            -- ^ input rows
                  -> Int            -- ^ input columns
                  -> Vector RealNum -- ^ bias
                  -> Matrix RealNum -- ^ kernel
                  -> Int            -- ^ kernel filters
                  -> Int            -- ^ kernel rows
                  -> Int            -- ^ kernel columns
                  -> Int            -- ^ stride height
                  -> Int            -- ^ stride width
                  -> Int            -- ^ output rows
                  -> Int            -- ^ output columns
                  -> Int            -- ^ pad left
                  -> Int            -- ^ pad top
                  -> Matrix RealNum -- ^ output of convolution
forwardBiasConv2d input channels rows cols bias kernel filters kernelRows kernelCols strideRows strideCols outRows outCols padLeft padTop =
  let outSize         = outRows * outCols * filters
  in  unsafePerformIO $ do
    let dataCol  = im2col_c channels rows cols kernelRows kernelCols strideRows strideCols padLeft padTop input outRows outCols
        gemmM    = LA.tr kernel LA.<> dataCol
        dataVid  = flatten gemmM
        (xPtr, _) = U.unsafeToForeignPtr0 dataVid
        (bPtr, _) = U.unsafeToForeignPtr0 bias

    withForeignPtr xPtr $ \xPtr' ->
      withForeignPtr bPtr $ \bPtr' ->
        in_place_add_per_channel_cpu xPtr' filters outRows outCols bPtr'

    let matVec = U.unsafeFromForeignPtr0 xPtr outSize
    return $ U.matrixFromVector U.RowMajor (outRows * filters) outCols matVec

-- | Efficient implementation of the backward pass for convolutions using the im2col trick as popularised by Caffe
backwardConv2d :: Matrix RealNum -- ^ input
               -> Int            -- ^ input channels
               -> Int            -- ^ input rows
               -> Int            -- ^ input columns
               -> Matrix RealNum -- ^ kernel
               -> Int            -- ^ kernel filters
               -> Int            -- ^ kernel rows
               -> Int            -- ^ kernel columns
               -> Int            -- ^ stride height
               -> Int            -- ^ stride width
               -> Int            -- ^ pad left
               -> Int            -- ^ pad top
               -> Int            -- ^ pad right
               -> Int            -- ^ pad bottom
               -> Matrix RealNum -- ^ derivative wrt out of layer
               -> Int            -- ^ output rows
               -> Int            -- ^ output columns
               -> (Matrix RealNum, Matrix RealNum)    -- ^ (derivated wrt input, derivative wrt kernel)
backwardConv2d input channels rows cols kernel filters kernelRows kernelCols strideRows strideCols padl padt padr padb dout outRows outCols =
  let dout'  = LA.reshape (outRows * outCols) $ LA.flatten dout
      dX_col = kernel LA.<> dout'
      rowCol = (div (rows + padt + padb - kernelRows) strideRows) + 1
      colCol = (div (cols + padl + padr - kernelCols) strideCols) + 1
      dX     = col2im_c channels rows cols kernelRows kernelCols strideRows strideCols padl padt rowCol colCol dX_col

      x_col  = im2col_c channels rows cols kernelRows kernelCols strideRows strideCols padl padt input outRows outCols
      dw_col = dout' LA.<> (LA.tr x_col)
      dw     = LA.tr $ reshapeMatrix filters (kernelRows * kernelCols * channels) dw_col 
  in (dX, dw) 

-- | Efficient implementation of the backward pass for convolutions using the im2col trick as popularised by Caffe
backwardBiasConv2d :: Matrix RealNum    -- ^ input
                   -> Int -- ^ input channels
                   -> Int -- ^ input rows
                   -> Int -- ^ input columns
                   -> Matrix RealNum -- ^ kernel
                   -> Int -- ^ kernel filters
                   -> Int -- ^ kernel rows
                   -> Int -- ^ kernel columns
                   -> Int -- ^ stride height
                   -> Int -- ^ stride width
                   -> Int -- ^ pad left
                   -> Int -- ^ pad top
                   -> Int -- ^ pad right
                   -> Int -- ^ pad bottom
                   -> Matrix RealNum -- ^ derivative wrt out of layer
                   -> Int -- ^ output rows
                   -> Int -- ^ output columns
                   -> (Matrix RealNum, Matrix RealNum, Vector RealNum)  -- ^ (derivated wrt input, derivative wrt kernel, derivative wrt bias)
backwardBiasConv2d input channels rows cols kernel filters kernelRows kernelCols strideRows strideCols padl padt padr padb dout outRows outCols =
  let (dX, dw) = backwardConv2d input channels rows cols kernel filters kernelRows kernelCols strideRows strideCols padl padt padr padb dout outRows outCols
      dB       = sum_over_channels_c dout filters outRows outCols
  in (dX, dw, dB)

sum_over_channels_c :: Matrix RealNum -> Int -> Int -> Int -> Vector RealNum
sum_over_channels_c mat channels rows cols = unsafePerformIO $ do
  let (xPtr, _) = U.unsafeToForeignPtr0 . flatten $ mat
  outPtr <- mallocForeignPtrArray channels

  withForeignPtr xPtr $ \xPtr' ->
    withForeignPtr outPtr $ \outPtr' ->
      sum_over_channels_cpu xPtr' channels rows cols outPtr'

  return $ U.unsafeFromForeignPtr0 outPtr channels

foreign import ccall unsafe
    sum_over_channels_cpu
      :: Ptr RealNum -> Int -> Int -> Int -> Ptr RealNum -> IO ()
