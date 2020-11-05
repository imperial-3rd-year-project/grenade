{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs     #-}

module Grenade.Utils.LinearAlgebra
    ( nsum
    , sumV
    , sumM
    , squareV
    , squareM
    ) where

import           GHC.TypeLits
import           Grenade.Types
import           Numeric.LinearAlgebra.Static
import           Numeric.LinearAlgebra        (sumElements)
import           Grenade.Core.Shape

sumV :: (KnownNat n) => R n -> RealNum
sumV v = fromDoubleToRealNum $ v <.> 1

sumM :: (KnownNat m, KnownNat n) => L m n -> RealNum
sumM m = fromDoubleToRealNum $ (m #> 1) <.> 1

squareV :: (KnownNat n) => R n -> R n
squareV v = dvmap (^ (2 :: Int)) v

squareM :: (KnownNat m, KnownNat n) => L m n ->  L m n
squareM m = dmmap (^ (2 :: Int)) m

-- | Helper function that sums the elements of a matrix
nsum :: S s -> Double
nsum (S1D x) = sumElements $ extract x
nsum (S2D x) = sumElements $ extract x
nsum (S3D x) = sumElements $ extract x
