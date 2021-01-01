{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE TypeApplications    #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.LRN where

import           Data.Singletons              ()
import           GHC.TypeLits
import           Hedgehog
import           Data.Proxy
import           Data.Kind (Type)
import           Test.Hedgehog.Hmatrix
import           Test.Grenade.Layers.Internal.Reference
import qualified System.Random.MWC as MWC
import           Control.DeepSeq
import           Control.Exception (evaluate)
import           Test.Hedgehog.Compat
import           Data.Serialize
import           Data.Either

import           Grenade.Core
import           Grenade.Layers.LRN
import           Grenade.Types
import           Numeric.LinearAlgebra.Data   as NLD hiding ((===))
import           Numeric.LinearAlgebra.Static as H hiding ((===))

expectedDepth1
  = [
      [0.99992500, 1.99940020, 2.99797659],
      [3.99520671, 4.99064546, 5.98385086],
      [6.97438480, 7.96181378, 8.94570965],
      [1.09990018, 2.09930569, 3.09776755],
      [4.09483851, 5.09007376, 6.08303166],
      [7.07327453, 8.06036937, 9.04388861],
      [1.19987041, 2.19920173, 3.19754459],
      [4.19445196, 5.18947928, 6.18218531],
      [7.17213277, 8.15888920, 9.14202759]
    ]

expectedDepth2
  = [
      [0.99992500, 1.99940020, 2.99797659],
      [3.99520671, 4.99064546, 5.98385086],
      [6.97438480, 7.96181378, 8.94570965],

      [1.09981771, 2.09867639, 3.09568020],
      [4.08993980, 5.08057535, 6.06671955],
      [7.04752047, 8.02214440, 8.98977845],

      [1.19976155, 2.19847498, 3.19524425],
      [4.18918085, 5.17940607, 6.16505399],
      [7.14527441, 8.11923556, 9.08612680]
    ]

expectedDepth3
  = [
      [0.99983428, 1.99873942, 2.99581955],
      [3.99018517, 4.98095623, 5.96726513],
      [6.94825962, 7.92310558, 8.89098967],

      [1.09969897, 2.09791554, 3.09330926],
      [4.08455479, 5.07034831, 6.04941408],
      [7.02051070, 7.98243693, 8.93403711],

      [1.19976155, 2.19847498, 3.19524425],
      [4.18918085, 5.17940607, 6.16505399],
      [7.14527441, 8.11923556, 9.08612680]
    ]

prop_lrn_forwards_fixed :: Property
prop_lrn_forwards_fixed = property $ do
  let ch1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      ch2 = map (map (0.1 +)) ch1
      ch3 = map (map (0.1 +)) ch2
      dat = concat [ch1, ch2, ch3]
      mat = (NLD.fromLists dat) :: NLD.Matrix RealNum

      lrn   = LRN :: (LRN "0.0001" "0.75" "1" 1)
      lrn'  = LRN :: (LRN "0.0001" "0.75" "1" 2)
      lrn'' = LRN :: (LRN "0.0001" "0.75" "1" 3)

      Just inp = (H.create mat) :: Maybe (H.L 9 3)
      (out,   expMat)   = getOutput lrn   expectedDepth1 inp
      (out',  expMat')  = getOutput lrn'  expectedDepth2 inp 
      (out'', expMat'') = getOutput lrn'' expectedDepth3 inp

  out   `isSimilarMatrixTo` expMat   
  out'  `isSimilarMatrixTo` expMat'  
  out'' `isSimilarMatrixTo` expMat'' 
  where
    -- Helper function to 
    getOutput :: (KnownSymbol a, KnownSymbol b, KnownSymbol k, KnownNat n) => LRN a b k n -> [[RealNum]] -> H.L 9 3 -> (Matrix RealNum, Matrix RealNum)
    getOutput l e inp = (H.extract o, NLD.fromLists e)
      where
      (_, S3D o :: (S ('D3 3 3 3))) = runForwards l ((S3D inp) :: (S ('D3 3 3 3)))

data OpaqueLRN :: Type where
  OpaqueLRN :: (KnownNat i) => LRN "0.0001" "0.75" "1" i -> OpaqueLRN

genOpaqueLRN :: Gen OpaqueLRN
genOpaqueLRN = do
  s :: Integer <- choose 1 7
  let Just s' = someNatVal s
  case s' of
      SomeNat (Proxy :: Proxy i') ->
          return . OpaqueLRN $ (LRN :: LRN "0.0001" "0.75" "1" i')

prop_lrn_forwards :: Property 
prop_lrn_forwards = property $ do
  OpaqueLRN (lrn :: LRN "0.0001" "0.75" "1" n) <- blindForAll genOpaqueLRN
  let n' = natVal (Proxy :: Proxy n)
  source <- forAll $ genLists3D 5 11 7 -- 5 channels, 11 rows, 7 columns
  let inp = S3D (H.fromList $ (concat . concat) source) :: S ('D3 11 7 5)
      (_, o :: (S ('D3 11 7 5))) = runForwards lrn inp
      out = naiveLRNForwards 0.0001 0.75 1 (fromIntegral n') source
  
  assert $ allClose o (S3D (H.fromList $ (concat . concat) out))

prop_lrn_backwards :: Property 
prop_lrn_backwards = property $ do
  OpaqueLRN (lrn :: LRN "0.0001" "0.75" "1" n) <- blindForAll genOpaqueLRN
  let n' = natVal (Proxy :: Proxy n)
  source <- forAll $ genLists3D 5 11 7 -- 5 channels, 11 rows, 7 columns
  let inp = S3D (H.fromList $ (concat . concat) source) :: S ('D3 11 7 5)
      (tape, o :: (S ('D3 11 7 5))) = runForwards lrn inp
      ((),    d :: (S ('D3 11 7 5))) = runBackwards lrn tape o
      out = naiveLRNForwards 0.0001 0.75 1 (fromIntegral n') source
      back = naiveLRNBackwards 0.0001 0.75 1 (fromIntegral n') source out

  assert $ allClose d (S3D (H.fromList $ (concat . concat) back))

prop_lrn_show :: Property
prop_lrn_show = withTests 1 $ property $ do
  let lrn = LRN :: (LRN "0.0001" "0.75" "1" 1)
  (show lrn) `seq` success

prop_lrn_rnf :: Property
prop_lrn_rnf = withTests 1 $ property $ do
  let lrn = LRN :: (LRN "0.0001" "0.75" "1" 1)
  (r :: ()) <- evalIO $ evaluate $ rnf lrn
  r `seq` success

prop_lrn_run_update :: Property
prop_lrn_run_update = withTests 1 $ property $ do
  source <- forAll $ genLists 12 17
  gen <- evalIO MWC.create
  (lrn :: LRN "0.0001" "0.75" "1" 1) <- evalIO $ createRandomWith UniformInit gen
  let mat = (NLD.fromLists source) :: NLD.Matrix RealNum
      Just inp = (H.create mat) :: Maybe (H.L 12 17)
      (tape, o :: (S ('D3 3 17 4))) = runForwards lrn ((S3D inp) :: (S ('D3 3 17 4)))
      (grad, _ :: (S ('D3 3 17 4))) = runBackwards lrn tape o
  (runUpdate defSGD lrn grad) `seq` (runUpdate defAdam lrn grad) `seq` success
  (reduceGradient @(LRN "0.0001" "0.75" "1" 1) [grad]) `seq` success

prop_lrn_serializable :: Property
prop_lrn_serializable = withTests 1 $ property $ do
  gen <- evalIO MWC.create
  (lrn :: LRN "0.0001" "0.75" "1" 1) <- evalIO $ createRandomWith UniformInit gen
  let bs  = encode lrn
      dec = decode bs :: Either String (LRN "0.0001" "0.75" "1" 1)
  assert $ isRight dec
  (put lrn) `seq` success

tests :: IO Bool
tests = checkParallel $$(discover)
