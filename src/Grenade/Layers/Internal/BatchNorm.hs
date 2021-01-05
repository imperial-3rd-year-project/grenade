{-# LANGUAGE ForeignFunctionInterface #-}
{-|
Module      : Grenade.Layers.Internal.Add
Description : Fast addition functions that call efficient C function
Maintainer  : Theo Charalambous
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Internal.BatchNorm (
    batchnorm
  ) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Vector, Matrix, flatten)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

-- | Efficient implementation of the forward propogation of batch normalization in testing mode.
batchnorm :: Int -> Int -> Int -> RealNum -> Matrix RealNum
             -> Vector RealNum -> Vector RealNum -> Vector RealNum -> Vector RealNum 
             -> Matrix RealNum
batchnorm channels rows cols epsilon m gamma beta running_mean running_var 
  = let outMatSize      = rows * cols * channels
        vec             = flatten m
    in unsafePerformIO $ do
      outPtr <- mallocForeignPtrArray outMatSize
      let (inPtr, _)    = U.unsafeToForeignPtr0 vec
          (gammaPtr, _) = U.unsafeToForeignPtr0 gamma
          (betaPtr, _)  = U.unsafeToForeignPtr0 beta
          (meanPtr, _)  = U.unsafeToForeignPtr0 running_mean
          (varPtr, _)   = U.unsafeToForeignPtr0 running_var
 
      withForeignPtr inPtr $ \inPtr' ->
        withForeignPtr gammaPtr $ \gammaPtr' ->
          withForeignPtr betaPtr $ \betaPtr' ->
            withForeignPtr meanPtr $ \meanPtr' ->
              withForeignPtr varPtr $ \varPtr' ->
                withForeignPtr outPtr $ \outPtr' ->
                  batchnorm_forward_cpu inPtr' gammaPtr' betaPtr' meanPtr' varPtr' epsilon channels rows cols outPtr'
    
      let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
      return (U.matrixFromVector U.RowMajor (rows * channels) cols matVec)
{-# INLINE batchnorm #-}

foreign import ccall unsafe
    batchnorm_forward_cpu
      :: Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> RealNum -> Int -> Int -> Int -> Ptr RealNum -> IO ()
