{-# LANGUAGE ForeignFunctionInterface #-}
{-|
Module      : Grenade.Layers.Internal.Activations
Description : Fast activation functions that call efficient C function
Maintainer  : Theo Charalambous
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Internal.Activations 
  ( relu
  , relu1d
  , leakyRelu
  , leakyRelu1d 
  ) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, Vector, flatten)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

-- | Apply ReLU on a matrix
relu :: Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
relu channels rows cols = leakyRelu channels rows cols 0
{-# INLINE relu #-}

-- | Apply ReLU on a vector
relu1d :: Int -> Vector RealNum -> Vector RealNum
relu1d size = leakyRelu1d size 0
{-# INLINE relu1d #-}

-- | Apply LeakyReLU on a matrix
leakyRelu :: Int -> Int -> Int -> RealNum -> Matrix RealNum -> Matrix RealNum
leakyRelu channels rows cols alpha m 
  = let vec = flatten m
        out = leakyRelu1d (channels * rows * cols) alpha vec
    in  U.matrixFromVector U.RowMajor (rows * channels) cols out
{-# INLINE leakyRelu #-}

-- | Apply LeakyReLU on a vector
leakyRelu1d :: Int -> RealNum -> Vector RealNum -> Vector RealNum
leakyRelu1d size alpha vec
  = unsafePerformIO $ do
      outPtr        <- mallocForeignPtrArray size
      let (inPtr, _) = U.unsafeToForeignPtr0 vec

      withForeignPtr inPtr $ \inPtr' ->
        withForeignPtr outPtr $ \outPtr' ->
          leaky_relu_forward inPtr' size alpha outPtr'
 
      return $ U.unsafeFromForeignPtr0 outPtr size
{-# INLINE leakyRelu1d #-}

foreign import ccall unsafe
   leaky_relu_forward
      :: Ptr RealNum -> Int -> RealNum -> Ptr RealNum -> IO ()
