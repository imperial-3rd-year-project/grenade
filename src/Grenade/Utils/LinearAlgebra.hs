{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds    #-}
{-# LANGUAGE GADTs        #-}
{-# LANGUAGE RankNTypes   #-}

module Grenade.Utils.LinearAlgebra
    ( bmean
    , bvar
    , vsqrt
    , vscale
    , vadd
    , vflatten
    , nsum
    , sumV
    , sumM
    , squareV
    , squareM
    , extractV
    ) where

import           Data.List                    (foldl1')
import           Data.Maybe                   (fromJust)
import           Data.Singletons
import           GHC.TypeLits
import           Grenade.Core.Shape
import           Grenade.Types
import           Numeric.LinearAlgebra        (flatten, sumElements, fromRows)
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

extractV :: S ('D1 x) -> R x
extractV (S1D v) = v

vscale :: KnownNat n => RealNum -> R n -> R n
vscale x v = dvmap (*x) v

vadd :: KnownNat n => RealNum -> R n -> R n
vadd x v = dvmap (+x) v

vsqrt :: KnownNat n => R n -> R n
vsqrt v = dvmap sqrt v

vflatten :: (KnownNat m, KnownNat n) => [R n] -> R m
vflatten xs = fromJust . create . flatten . fromRows $ map extract xs

-- bsqrt :: SingI s => [S s] -> [S s]
-- bsqrt xs = map msqrt xs

-- msqrt :: SingI s => S s -> S s
-- msqrt (S1D x) = S1D $ dvmap sqrt x
-- msqrt (S2D x) = S2D $ dmmap sqrt x
-- msqrt (S3D x) = S3D $ dmmap sqrt x

-- mscale :: SingI s => RealNum -> S s -> S s
-- mscale r (S1D x) = S1D $ dvmap (* r) x
-- mscale r (S2D x) = S2D $ dmmap (* r) x
-- mscale r (S3D x) = S3D $ dmmap (* r) x

-- madd :: SingI s => RealNum -> S s -> S s
-- madd r (S1D x) = S1D $ dvmap (+ r) x
-- madd r (S2D x) = S2D $ dmmap (+ r) x
-- madd r (S3D x) = S3D $ dmmap (+ r) x
