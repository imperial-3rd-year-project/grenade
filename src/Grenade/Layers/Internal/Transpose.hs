{-# LANGUAGE ForeignFunctionInterface #-}
{-|
Module      : Grenade.Layers.Internal.Transpose
Description : Functions to quickly transpose a matrix
Maintainer  : Theo Charalambous
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Internal.Transpose (
  transpose4d
) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray,
                                              withForeignPtr)
import           Foreign.C.Types             (CInt)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, Vector, cmap, flatten)
import qualified Numeric.LinearAlgebra.Data  as D
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

-- | Transpose a 4d matrix, similarly to the tranpose function in numpy
transpose4d :: [Int]           -- ^ original dimensions
            -> Vector RealNum  -- ^ permutation vector
            -> Matrix RealNum  -- ^ input matrix
            -> Matrix RealNum  -- ^ transposed matrix
transpose4d dims@[n, c, h, w] permsV m
  = let outMatSize = n * c * h * w
        outW       = dims !! (round $ permsV D.! 3)
        vec        = flatten m
        permsV'    = cmap (U.fi . round) permsV  :: Vector CInt
        dimsV      = D.fromList $ map U.fi dims  :: Vector CInt
    in unsafePerformIO $ do
      outPtr        <- mallocForeignPtrArray outMatSize
      let (inPtr, _) = U.unsafeToForeignPtr0 vec
          (pPtr, _)  = U.unsafeToForeignPtr0 permsV'
          (dPtr, _)  = U.unsafeToForeignPtr0 dimsV

      withForeignPtr inPtr $ \inPtr' ->
        withForeignPtr pPtr $ \pPtr' ->
          withForeignPtr dPtr $ \dPtr' ->
            withForeignPtr outPtr $ \outPtr' ->
              transpose_4d inPtr' dPtr' pPtr' outPtr'

      let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
      return (U.matrixFromVector U.RowMajor (div outMatSize outW) outW matVec)
{-# INLINE transpose4d #-}

foreign import ccall unsafe
    transpose_4d
      :: Ptr RealNum -> Ptr CInt -> Ptr CInt -> Ptr RealNum -> IO ()
