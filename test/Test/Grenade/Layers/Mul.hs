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

module Test.Grenade.Layers.Mul where

import           Data.Constraint
import           Data.Maybe                   (fromJust)
import           Data.Proxy
import           Unsafe.Coerce

import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Layers.Mul
import           Grenade.Types

import           Numeric.LinearAlgebra        hiding (R, konst, randomVector,
                                               uniformSample, (===))
import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Data   as D
import           Numeric.LinearAlgebra.Static (L, R)
import qualified Numeric.LinearAlgebra.Static as H

import           Hedgehog
import qualified Hedgehog.Gen                 as Gen
import qualified Hedgehog.Range               as Range

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

prop_mul_scalar_one_does_nothing = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100
  channels :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c))) -> do
      input :: S ('D3 h w c) <- forAll genOfShape

      let layer    = initMul :: Mul 1 1 1
          S3D out  = snd $ runForwards layer input :: S ('D3 h w c)
          S3D inp' = input :: S ('D3 h w c)

      H.extract inp' === H.extract out

prop_mul_random_scalar_as_expected = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100
  channels :: Int <- forAll $ choose 2 100

  scalar :: R 1 <- forAll randomVector
  let scale = H.extract scalar LA.! 0

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c))) -> do
      input :: S ('D3 h w c) <- forAll genOfShape

      let layer    = Mul scalar :: Mul 1 1 1
          S3D out  = snd $ runForwards layer input :: S ('D3 h w c)
          inp' = (\(S3D x) -> H.dmmap (* scale) x) input :: L (h * c) w

      H.extract inp' === H.extract out

tests :: IO Bool
tests = checkParallel $$(discover)
