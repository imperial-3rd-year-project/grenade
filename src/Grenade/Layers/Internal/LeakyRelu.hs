{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.LeakyRelu (
  applyLeakyReluBulk 
) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, flatten)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

applyLeakyReluBulk :: Int -> Int -> Int -> RealNum -> Matrix RealNum -> Matrix RealNum
applyLeakyReluBulk channels rows cols alpha m
  = let outMatSize      = rows * cols * channels
        vec             = flatten m
    in unsafePerformIO $ do
      outPtr        <- mallocForeignPtrArray outMatSize
      let (inPtr, _) = U.unsafeToForeignPtr0 vec

      withForeignPtr inPtr $ \inPtr' ->
          withForeignPtr outPtr $ \outPtr' ->
            apply_leaky_relu_bulk inPtr' channels rows cols alpha outPtr'
 
      let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
      return (U.matrixFromVector U.RowMajor (rows * channels) cols matVec)
{-# INLINE applyLeakyReluBulk #-}

foreign import ccall unsafe
   apply_leaky_relu_bulk
      :: Ptr RealNum -> Int -> Int -> Int -> RealNum -> Ptr RealNum -> IO ()
