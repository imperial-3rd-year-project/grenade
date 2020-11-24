{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.LRN where

import           Data.Singletons               ()
import           GHC.TypeLits
import           Hedgehog
import           Test.Hedgehog.Hmatrix

import           Grenade.Core
import           Grenade.Layers.LRN
import           Numeric.LinearAlgebra.Data as NLD hiding ((===))
import           Numeric.LinearAlgebra.Static as S hiding ((===))

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

getOutput :: (KnownSymbol a, KnownSymbol b, KnownSymbol k, KnownNat n) => LRN a b k n -> [[Double]] -> S.L 9 3 -> (S ('D3 3 3 3), S ('D3 3 3 3))
getOutput l e inp = (o, (S3D (eMat :: S.L 9 3)))
  where
  (_, o :: (S ('D3 3 3 3))) = runForwards l ((S3D inp) :: (S ('D3 3 3 3)))
  Just eMat = S.create $ NLD.fromLists e


prop_lrn_forwards :: Property
prop_lrn_forwards = property $ do
  let ch1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      ch2 = map (map (0.1 +)) ch1
      ch3 = map (map (0.1 +)) ch2
      dat = concat [ch1, ch2, ch3]
      mat = (NLD.fromLists dat) :: NLD.Matrix Double
      lrn   = LRN :: (LRN "0.0001" "0.75" "1" 1)
      lrn'  = LRN :: (LRN "0.0001" "0.75" "1" 2)
      lrn'' = LRN :: (LRN "0.0001" "0.75" "1" 3)
      Just inp = (S.create mat) :: Maybe (S.L 9 3)
      (out,   expMat)   = getOutput lrn   expectedDepth1 inp
      (out',  expMat')  = getOutput lrn'  expectedDepth2 inp 
      (out'', expMat'') = getOutput lrn'' expectedDepth3 inp
  assert $ allCloseP out   expMat   0.000001
  assert $ allCloseP out'  expMat'  0.000001
  assert $ allCloseP out'' expMat'' 0.000001


tests :: IO Bool
tests = checkParallel $$(discover)

