{-# OPTIONS_GHC -Wno-incomplete-patterns #-}

{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}

module Test.Grenade.Utils.LinearAlgebra where

import           Data.List                    (foldl1')
import           Data.Proxy

import           GHC.TypeLits

import           Hedgehog
import qualified Hedgehog.Gen                 as Gen
import qualified Hedgehog.Range               as Range

import qualified Numeric.LinearAlgebra.Static as NLA

import           Grenade
import           Grenade.Utils.LinearAlgebra
import           Test.Hedgehog.Hmatrix

calcMean :: [[Double]] -> [Double]
calcMean xs = let l = fromIntegral $ length xs :: Double
              in  map (/ l) $ foldl1' (zipWith (+)) xs

calcVariance :: [[Double]] -> [Double]
calcVariance xs = let l  = fromIntegral $ length xs :: Double
                      ms = calcMean xs 
                      ys = map (map (^(2 :: Int)) . zipWith (-) ms) xs
                  in  map (/ l) $ foldl1' (zipWith (+)) ys

prop_1D_mean :: Property
prop_1D_mean = property $ do
  batches   <- forAll $ Gen.int $ Range.linear 2 100
  vecLength <- forAll $ Gen.int $ Range.linear 2 100
  xs        <- forAll $ genLists batches vecLength

  case someNatVal $ fromIntegral vecLength of 
    Just (SomeNat (Proxy :: Proxy v)) -> 
      let xsVer   = calcMean xs
          xs'     = extractVec (bmean $ map (S1D . NLA.fromList) xs :: S ('D1 v))
      in xsVer === xs'

prop_2D_mean :: Property
prop_2D_mean = property $ do
  batches   <- forAll $ Gen.int $ Range.linear 2 100
  matHeight <- forAll $ Gen.int $ Range.linear 2 100
  matWidth  <- forAll $ Gen.int $ Range.linear 2 100
  xs        <- forAll $ genLists batches (matHeight * matWidth)

  case (someNatVal $ fromIntegral matHeight, someNatVal $ fromIntegral matWidth) of 
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w))) -> 
      let xsVer    = calcMean xs
          mats     = map (S2D . NLA.fromList) xs :: [S ('D2 h w)]
          xs'      = bmean mats
      in xsVer === (concat $ extractMat xs')

prop_1D_variance :: Property
prop_1D_variance = property $ do
  batches   <- forAll $ Gen.int $ Range.linear 2 100
  vecLength <- forAll $ Gen.int $ Range.linear 2 100
  xs        <- forAll $ genLists batches vecLength

  case someNatVal $ fromIntegral vecLength of 
    Just (SomeNat (Proxy :: Proxy v)) -> 
      let xsReference = S1D . NLA.fromList $ calcVariance xs :: S ('D1 v)
          xs'         = bvar $ map (S1D . NLA.fromList) xs :: S ('D1 v)
      in assert $ allClose xsReference xs'

prop_2D_variance :: Property
prop_2D_variance = property $ do
  batches   <- forAll $ Gen.int $ Range.linear 2 100
  matHeight <- forAll $ Gen.int $ Range.linear 2 100
  matWidth  <- forAll $ Gen.int $ Range.linear 2 100
  xs        <- forAll $ genLists batches (matHeight * matWidth)

  case (someNatVal $ fromIntegral matHeight, someNatVal $ fromIntegral matWidth) of 
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w))) -> 
      let xsReference = S2D . NLA.fromList $ calcVariance xs :: S ('D2 h w)
          mats        = map (S2D . NLA.fromList) xs :: [S ('D2 h w)]
          xs'         = bvar mats
      in assert $ allClose xsReference xs'

tests :: IO Bool
tests = checkParallel $$(discover)
