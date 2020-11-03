{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Utils.LinearAlgebra
    ( bmean
    , bvar
    , bvar'
    , vsqrt
    , msqrt
    , sreshape
    , sflatten
    , vscale
    , vadd
    , vflatten
    , nsum
    , sumV
    , sumM
    , squareV
    , squareM
    , extractV
    , extractM2D
    , batchNormMean
    , batchNormVariance
    , vectorToList
    , listToVector
    ) where

import           Data.List                    (foldl', foldl1')
import           Data.Maybe                   (fromJust)
import           Data.Proxy
import           Data.Singletons
import           GHC.TypeLits
import           Grenade.Core.Shape
import           Grenade.Types
import           Numeric.LinearAlgebra        (flatten, fromRows, sumElements, fromList)
import           Numeric.LinearAlgebra.Data   as D hiding (L, R)
import           Numeric.LinearAlgebra.Static hiding (fromList)

sumV :: (KnownNat n) => R n -> RealNum
sumV v = fromDoubleToRealNum $ v <.> 1

sumM :: (KnownNat m, KnownNat n) => L m n -> RealNum
sumM m = fromDoubleToRealNum $ (m #> 1) <.> 1

squareV :: (KnownNat n) => R n -> R n
squareV = dvmap (^ (2 :: Int))

squareM :: (KnownNat m, KnownNat n) => L m n ->  L m n
squareM = dmmap (^ (2 :: Int))

-- | Helper function that sums the elements of a matrix
nsum :: S s -> Double
nsum (S1D x) = sumElements $ extract x
nsum (S2D x) = sumElements $ extract x
nsum (S3D x) = sumElements $ extract x

-- | Calculate elementwise mean of list of matrices
bmean :: forall s. SingI s => [S s] -> S s
bmean xs
  = let (!m, !l) = foldl' (\(m, l) x -> (m + x, l + 1)) (0 :: S s, 0) xs :: (S s, Int)
    in  case (m, l) of
          (S1D x, _) -> S1D $ dvmap (/ fromIntegral l) x
          (S2D x, _) -> S2D $ dmmap (/ fromIntegral l) x
          (S3D x, _) -> S3D $ dmmap (/ fromIntegral l) x

-- | Calculate element wise variance
bvar :: forall s. SingI s => [S s] -> S s
bvar xs
  = let !m       = bmean xs
        (!v, !l) = foldl' (\(v, l) x -> (v + (x - m)**2, l + 1)) (0 :: S s, 0) xs :: (S s, Int)
    in  case (v, l) of
          (S1D x, _) -> S1D $ dvmap (/ fromIntegral l) x
          (S2D x, _) -> S2D $ dmmap (/ fromIntegral l) x
          (S3D x, _) -> S3D $ dmmap (/ fromIntegral l) x

-- | Calculate element wise variance, with the mean precalculated
bvar' :: forall s. SingI s => S s -> [S s] -> S s
bvar' m xs
  = let (!v, !l) = foldl' (\(v, l) x -> (v + (x - m)**2, l + 1)) (0 :: S s, 0) xs :: (S s, Int)
    in  case (v, l) of
          (S1D x, _) -> S1D $ dvmap (/ fromIntegral l) x
          (S2D x, _) -> S2D $ dmmap (/ fromIntegral l) x
          (S3D x, _) -> S3D $ dmmap (/ fromIntegral l) x

batchNormMean :: forall n. KnownNat n => [R n] -> Double
batchNormMean vs 
  = let hw  = fromIntegral $ natVal (Proxy :: Proxy n)
        vs' = map (sumElements . extract) vs :: [Double]
        n   = fromIntegral $ length vs 
    in  sum vs' / (hw * n)

batchNormVariance :: forall n. KnownNat n => [R n] -> Double
batchNormVariance vs 
  = let mu  = batchNormMean vs
        hw  = fromIntegral $ natVal (Proxy :: Proxy n)
        vs' = map (\x -> sumElements . extract $ dvmap (\a -> (a - mu) ^ 2) x) vs :: [Double] 
        n   = fromIntegral $ length vs 
    in  sum vs' / (hw * n)

extractV :: S ('D1 x) -> R x
extractV (S1D v) = v

extractM2D :: S ('D2 x y) -> L x y
extractM2D (S2D m) = m

vscale :: KnownNat n => RealNum -> R n -> R n
vscale x v = dvmap (*x) v

vadd :: KnownNat n => RealNum -> R n -> R n
vadd x v = dvmap (+x) v

vsqrt :: KnownNat n => R n -> R n
vsqrt v = dvmap sqrt v

msqrt :: (KnownNat n, KnownNat m) => L m n -> L m n
msqrt m = dmmap sqrt m

vflatten :: (KnownNat m, KnownNat n) => [R n] -> R m
vflatten xs = fromJust . create . flatten . fromRows $ map extract xs

sflatten :: (KnownNat m, KnownNat n) => L m n -> R (m * n)
sflatten = fromJust . create . flatten . extract

sreshape :: forall m n. (KnownNat m, KnownNat n) => R (m * n) -> L m n
sreshape v
  = let rows = fromIntegral $ natVal (Proxy :: Proxy m)
    in  fromJust . create . D.reshape rows . extract $ v


vectorToList :: KnownNat n => R n -> [RealNum]
vectorToList = toList . extract

listToVector :: KnownNat n => [RealNum] -> R n
listToVector = fromJust . create . fromList

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
