{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ViewPatterns     #-}


module Grenade.Layers.Internal.Hmatrix (
    reshapeMatrix
  ) where

import           Numeric.LinearAlgebra
import           Numeric.LinearAlgebra.Devel

-- | Reshape a row major matrix
reshapeMatrix :: (Element t, Num t, Container Vector t) => Int -> Int -> Matrix t -> Matrix t
reshapeMatrix r c m@(size->(r', c'))
    | r * c == r' * c' = matrixFromVector RowMajor r c $ flatten m
    | otherwise        = error $ "can't reshape matrix of shape dim = " ++ show (r', c') ++ " to matrix of shape " ++ show (r, c)
{-# INLINE reshapeMatrix #-}