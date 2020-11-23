{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Add (
  addPerChannel
) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, Vector, flatten)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

addPerChannel :: Int -> Int -> Int -> Matrix RealNum -> Vector RealNum -> Matrix RealNum
addPerChannel channels rows cols m b
  = let outMatSize      = rows * cols * channels
        vec             = flatten m
    in unsafePerformIO $ do
      outPtr        <- mallocForeignPtrArray outMatSize
      let (inPtr, _) = U.unsafeToForeignPtr0 vec
          (bPtr, _)  = U.unsafeToForeignPtr0 b
 
      withForeignPtr inPtr $ \inPtr' ->
        withForeignPtr bPtr $ \bPtr' ->
          withForeignPtr outPtr $ \outPtr' ->
            add_per_channel_cpu inPtr' channels rows cols bPtr' outPtr'
 
      let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
      return (U.matrixFromVector U.RowMajor (rows * channels) cols matVec)
{-# INLINE addPerChannel #-}

foreign import ccall unsafe
    add_per_channel_cpu
      :: Ptr RealNum -> Int -> Int -> Int -> Ptr RealNum -> Ptr RealNum -> IO ()
