{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE TypeApplications    #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.LRN where

import           Data.Singletons               ()
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
import           Numeric.LinearAlgebra.Data as NLD hiding ((===))
import           Numeric.LinearAlgebra.Static as H hiding ((===))

expectedDepth1
  = [
      [0.99992500656189853281, 1.99940020992302902592, 2.99797659337299204907],
      [3.99520671015876160936, 4.99064546092531813315, 5.98385086216740891274],
      [6.97438480279017181118, 7.96181378595566702217, 8.94570965349002911182],

      [1.09990018556779967085, 2.09930569291082669281, 3.09776755213446852721],
      [4.09483851636593687573, 5.09007376342854911400, 6.08303166323381461211],
      [7.07327453299620856342, 8.06036937745664872068, 9.04388861140008692985],

      [1.19987041632744495523, 2.19920173805711716142, 3.19754459994463013928],
      [4.19445196276661480539, 5.18947928920103684902, 6.18218531013063721247],
      [7.17213277848370900358, 8.15888920780734672178, 9.14202759286949095952]
    ]

expectedDepth2 
  = [
      [0.99992500656189853281, 1.99940020992302902592, 2.99797659337299204907],
      [3.99520671015876160936, 4.99064546092531813315, 5.98385086216740891274],
      [6.97438480279017181118, 7.96181378595566702217, 8.94570965349002911182],

      [1.09981771024995600428, 2.09867639897079083511, 3.09568020869522664285],
      [4.08993980262724576846, 5.08057535608961430285, 6.06671955060380518887],
      [7.04752047345527632416, 8.02214440184233801290, 8.98977845221027926925],

      [1.19976155528875683132, 2.19847498426206255928, 3.19524425944437373559],
      [4.18918085827706043744, 5.17940607335874148021, 6.16505399805195786200],
      [7.14527441542452734780, 8.11923556976295479615, 9.08612680142306139430]
    ]

expectedDepth3
  = [
      [0.99983428204541446860, 1.99873942759122913415, 2.99581955680183220636],
      [3.99018517329487387713, 4.98095623146040722418, 5.96726513174144823637],
      [6.94825962171646960286, 7.92310558206650750179, 8.89098967801016648593],

      [1.09969897113963055446, 2.09791554153570025676, 3.09330926290207530727],
      [4.08455479164546808590, 5.07034831189770862636, 6.04941408910535827204],
      [7.02051070809165356224, 7.98243693066330806118, 8.93403711513257192678],

      [1.19976155528875683132, 2.19847498426206255928, 3.19524425944437373559],
      [4.18918085827706043744, 5.17940607335874148021, 6.16505399805195786200],
      [7.14527441542452734780, 8.11923556976295479615, 9.08612680142306139430]
    ]

prop_lrn_forwards_fixed :: Property
prop_lrn_forwards_fixed = property $ do
  let ch1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      ch2 = map (map (0.1 +)) ch1
      ch3 = map (map (0.1 +)) ch2
      dat = concat [ch1, ch2, ch3]
      mat = (NLD.fromLists dat) :: NLD.Matrix Double
      lrn   = LRN :: (LRN "0.0001" "0.75" "1" 1)
      lrn'  = LRN :: (LRN "0.0001" "0.75" "1" 2)
      lrn'' = LRN :: (LRN "0.0001" "0.75" "1" 3)
      Just inp = (H.create mat) :: Maybe (H.L 9 3)
      (out,   expMat)   = getOutput lrn   expectedDepth1 inp
      (out',  expMat')  = getOutput lrn'  expectedDepth2 inp 
      (out'', expMat'') = getOutput lrn'' expectedDepth3 inp
  assert $ allCloseP out   expMat   0.000001
  assert $ allCloseP out'  expMat'  0.000001
  assert $ allCloseP out'' expMat'' 0.000001
  where
    -- Helper function to 
    getOutput :: (KnownSymbol a, KnownSymbol b, KnownSymbol k, KnownNat n) => LRN a b k n -> [[Double]] -> H.L 9 3 -> (S ('D3 3 3 3), S ('D3 3 3 3))
    getOutput l e inp = (o, (S3D (eMat :: H.L 9 3)))
      where
      (_, o :: (S ('D3 3 3 3))) = runForwards l ((S3D inp) :: (S ('D3 3 3 3)))
      Just eMat = H.create $ NLD.fromLists e


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
  let mat = (NLD.fromLists source) :: NLD.Matrix Double
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
