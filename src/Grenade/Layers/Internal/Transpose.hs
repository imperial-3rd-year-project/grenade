{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Transpose (
  transpose4d
) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray,
                                              withForeignPtr)
import           Foreign.C.Types             (CInt)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, Vector, cmap, flatten,
                                              vector)
import qualified Numeric.LinearAlgebra.Data  as D
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

transpose4d :: [Int] -> Vector Double -> Matrix RealNum -> Matrix RealNum
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
