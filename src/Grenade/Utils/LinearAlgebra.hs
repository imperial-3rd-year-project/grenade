{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds    #-}
{-# LANGUAGE GADTs        #-}

module Grenade.Utils.LinearAlgebra
    ( bmean
    , bvar
    , nsum
    , sumV
    , sumM
    , squareV
    , squareM
    ) where

import           Data.List                    (foldl1')
import           Data.Singletons
import           GHC.TypeLits
import           Grenade.Core.Shape
import           Grenade.Types
import           Numeric.LinearAlgebra        (sumElements)
import           Numeric.LinearAlgebra.Static

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

-- | Calculate elementwise mean of list of matrices
bmean :: SingI s => [S s] -> S s
bmean xs
  = let !l = fromIntegral $ length xs :: Double
    in  case foldl1' (+) xs of
          S1D x -> S1D $ dvmap (/ l) x
          S2D x -> S2D $ dmmap (/ l) x
          S3D x -> S3D $ dmmap (/ l) x

-- | Calculate element wise variance
bvar :: SingI s => [S s] -> S s
bvar xs
  = let !l = fromIntegral $ length xs :: Double
        !m = bmean xs
    in  case foldl1' (+) $ map (\x -> (x - m) ** 2) xs of
          S1D x -> S1D $ dvmap (/ l) x
          S2D x -> S2D $ dmmap (/ l) x
          S3D x -> S3D $ dmmap (/ l) x
