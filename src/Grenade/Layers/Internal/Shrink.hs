{-# LANGUAGE ForeignFunctionInterface #-}
{-|
Module      : Grenade.Layers.Internal.Shrink
Description : Functions to shrink a matrix
Maintainer  : Theo Charalambous
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Internal.Shrink (
    shrink_2d,
    shrink_2d_rgba
  ) where

import qualified Data.Vector.Storable        as S

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, flatten)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Data.Word

import           Grenade.Types

shrink_2d :: Int            -- ^ input rows
          -> Int            -- ^ input cols
          -> Int            -- ^ target rows
          -> Int            -- ^ target cols
          -> Matrix RealNum -- ^ input matrix
          -> Matrix RealNum -- ^ shrinked matrix 
shrink_2d rows cols rows' cols' m
 = let outMatSize      = rows' * cols'
       vec             = flatten m
   in unsafePerformIO $ do
     --print vec
     outPtr           <- mallocForeignPtrArray outMatSize
     let (inPtr, _, _) = S.unsafeToForeignPtr vec

     withForeignPtr inPtr $ \inPtr' ->
        withForeignPtr outPtr $ \outPtr' ->
          shrink_2d_cpu inPtr' rows cols rows' cols' outPtr'

     let matVec = S.unsafeFromForeignPtr0 outPtr outMatSize
     return (U.matrixFromVector U.RowMajor rows' cols' matVec)
--{-# INLINE shrink_2d #-}

shrink_2d_rgba :: Int            -- ^ input rows
               -> Int            -- ^ input cols
               -> Int            -- ^ target rows
               -> Int            -- ^ target cols
               -> S.Vector Word8 -- ^ vector of word8s in the form rgba, like from a HTML canvas
               -> Matrix RealNum -- shrinked matrix
shrink_2d_rgba rows cols rows' cols' vec
  = let outMatSize = rows' * cols'
    in unsafePerformIO $ do 
      outPtr           <- mallocForeignPtrArray outMatSize
      let (inPtr, _, _) = S.unsafeToForeignPtr vec

      -- print vec

      withForeignPtr inPtr $ \inPtr' ->
        withForeignPtr outPtr $ \outPtr' ->
          shrink_2d_rgba_cpu inPtr' rows cols rows' cols' outPtr'
      
      let matVec = S.unsafeFromForeignPtr0 outPtr outMatSize
      return (U.matrixFromVector U.RowMajor rows' cols' matVec)
--{-# INLINE shrink_2d_rgba #-}

foreign import ccall unsafe
    shrink_2d_cpu :: Ptr RealNum -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()

foreign import ccall unsafe
    shrink_2d_rgba_cpu :: Ptr Word8 -> Int -> Int -> Int -> Int -> Ptr RealNum -> IO ()
