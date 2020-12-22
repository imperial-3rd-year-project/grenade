{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Convolution (
    im2col
  , im2col_new
  , im2col_memcpy
  , col2im
  , col2vid
  , vid2col
  ) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, cols, flatten, rows)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

col2vid :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
col2vid kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = cols dataCol `div` (kernelRows * kernelColumns)
  in  col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
col2im kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = 1
  in  col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol

col2im_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol =
  let vec = flatten dataCol
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (height * width * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        col2im_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (height * width * channels)
    return $ U.matrixFromVector U.RowMajor (height * channels) width matVec

foreign import ccall unsafe
    col2im_cpu
      :: Ptr RealNum -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()

vid2col :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
vid2col kernelRows kernelColumns strideRows strideColumns height width dataVid =
  let channels = rows dataVid `div` height
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataVid


im2col :: Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2col kernelRows kernelColumns strideRows strideColumns dataIm =
  let channels = 1
      height = rows dataIm
      width  = cols dataIm
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataIm

im2col_new :: Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2col_new kernelRows kernelColumns strideRows strideColumns dataIm =
  let channels = 1
      height = rows dataIm
      width  = cols dataIm
  in  im2col_c_new channels height width kernelRows kernelColumns strideRows strideColumns dataIm

im2col_memcpy :: Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2col_memcpy kernelRows kernelColumns strideRows strideColumns dataIm =
  let channels = 1
      height = rows dataIm
      width  = cols dataIm
  in  im2col_c_memcpy channels height width kernelRows kernelColumns strideRows strideColumns dataIm

im2col_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataIm =
  let vec    = flatten dataIm
      rowOut = (height - kernelRows) `div` strideRows + 1
      colOut = (width - kernelColumns) `div` strideColumns + 1
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = rowOut * colOut
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (numberOfPatches * kernelSize * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        im2col_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (numberOfPatches * kernelSize * channels)
    return $ U.matrixFromVector U.RowMajor numberOfPatches (kernelSize * channels) matVec
    
im2col_c_new :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2col_c_new channels height width kernelRows kernelColumns strideRows strideColumns dataIm =
  let vec    = flatten dataIm
      rowOut = (height - kernelRows) `div` strideRows + 1
      colOut = (width - kernelColumns) `div` strideColumns + 1
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = rowOut * colOut
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (numberOfPatches * kernelSize * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        im2col_cpu_new inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (numberOfPatches * kernelSize * channels)
    return $ U.matrixFromVector U.RowMajor numberOfPatches (kernelSize * channels) matVec

im2col_c_memcpy :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2col_c_memcpy channels height width kernelRows kernelColumns strideRows strideColumns dataIm =
  let vec    = flatten dataIm
      rowOut = (height - kernelRows) `div` strideRows + 1
      colOut = (width - kernelColumns) `div` strideColumns + 1
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = rowOut * colOut
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (numberOfPatches * kernelSize * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        im2col_cpu_memcpy inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (numberOfPatches * kernelSize * channels)
    return $ U.matrixFromVector U.RowMajor numberOfPatches (kernelSize * channels) matVec

foreign import ccall unsafe
    im2col_cpu
      :: Ptr RealNum -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()

foreign import ccall unsafe
    im2col_cpu_new
      :: Ptr RealNum -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()

foreign import ccall unsafe
    im2col_cpu_memcpy
      :: Ptr RealNum -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()
