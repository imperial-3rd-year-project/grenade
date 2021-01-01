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

module Test.Grenade.Layers.LeakyRelu where

import           Data.Proxy

import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Layers.LeakyRelu

import qualified Numeric.LinearAlgebra        as LA
import           Numeric.LinearAlgebra.Static (L)
import qualified Numeric.LinearAlgebra.Static as H

import           Hedgehog
import qualified Hedgehog.Range               as Range

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

prop_leaky_relu_correct_on_one_element = property $ do
  input :: S ('D1 1) <- forAll genOfShape
  alpha              <- forAll $ genRealNum (Range.constant 0 0.9)

  let layer   = LeakyRelu alpha :: LeakyRelu
      S1D out = snd $ runForwards layer input :: S ('D1 1)
      x       = (\(S1D a) -> H.extract a LA.! 0) input
  
  H.extract out LA.! 0 === if x >= 0 then x else x * alpha

prop_leaky_relu_correct_on_3D_array = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100
  channels :: Int <- forAll $ choose 2 100
  alpha           <- forAll $ genRealNum (Range.constant 0 0.9)

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c))) -> do
      input :: S ('D3 h w c) <- forAll genOfShape

      let layer    = LeakyRelu alpha :: LeakyRelu
          S3D out  = snd $ runForwards layer input :: S ('D3 h w c)
          inp' = (\(S3D x) -> H.dmmap (\a -> if a >= 0 then a else a * alpha) x) input :: L (h * c) w

      H.extract inp' === H.extract out

tests :: IO Bool
tests = checkParallel $$(discover)
