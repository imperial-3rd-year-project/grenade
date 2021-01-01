{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}

{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}

module Test.Grenade.Layers.Add where

import           Data.Constraint
import           Data.Maybe                   (fromJust)
import           Data.Proxy
import           Unsafe.Coerce

import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Layers.Add

import           Numeric.LinearAlgebra        hiding (R, konst, randomVector,
                                               uniformSample, (===), reshape)
import qualified Numeric.LinearAlgebra.Data   as D
import           Numeric.LinearAlgebra.Static (L, R)
import qualified Numeric.LinearAlgebra.Static as H

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

prop_add_zero_does_nothing = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100
  channels :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c))) -> do
      input :: S ('D3 h w c) <- forAll genOfShape

      let layer    = initAdd :: Add c 1 1
          S3D out  = snd $ runForwards layer input :: S ('D3 h w c)
          S3D inp' = input :: S ('D3 h w c)

      H.extract inp' === H.extract out

prop_add_per_channel_words_as_expected = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w))) ->
      case ( unsafeCoerce (Dict :: Dict ()) :: Dict ( (h * 3) ~ (h + h + h) ) ) of
        Dict -> do
          channel_one   :: R (h * w) <- forAll randomVector
          channel_two   :: R (h * w) <- forAll randomVector
          channel_three :: R (h * w) <- forAll randomVector
          bias          :: R 3       <- forAll randomVector

          let reshape = fromJust . H.create . (D.reshape width) . H.extract
              c1      = reshape channel_one   :: L h w
              c2      = reshape channel_two   :: L h w
              c3      = reshape channel_three :: L h w

              bias' = H.extract bias

              o1 = H.dmmap (\a -> a + bias' ! 0) c1 :: L h w
              o2 = H.dmmap (\a -> a + bias' ! 1) c2 :: L h w
              o3 = H.dmmap (\a -> a + bias' ! 2) c3 :: L h w

              out = o1 H.=== o2 H.=== o3 :: L (h * 3) w
              inp = c1 H.=== c2 H.=== c3 :: L (h * 3) w

              layer    = Add bias :: Add 3 1 1
              inp'     = S3D inp :: S ('D3 h w 3)
              S3D out' = snd $ runForwards layer inp' :: S ('D3 h w 3)

          H.extract out === H.extract out'

tests :: IO Bool
tests = checkParallel $$(discover)
