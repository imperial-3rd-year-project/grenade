{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

module Test.Grenade.Layers.Trivial where

import           GHC.TypeLits
import           Grenade.Layers.Trivial
import           Grenade.Core
import           Data.Proxy

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

import           Numeric.LinearAlgebra.Static as H hiding ((===), create)
import           Data.Serialize
import           Data.Either
import           System.Random.MWC             (create)

prop_trivial_run_forward_passes_along_input :: Property
prop_trivial_run_forward_passes_along_input = property $ do
  input :: S ('D3 5 7 3) <- forAll genOfShape

  let layer = Trivial
      ((), out) = runForwards layer input
      ((), y)   = runBackwards layer () out
      S3D input' = input
      S3D out'   = out
      S3D y'     = y
  H.extract out' === H.extract input'
  H.extract out' === H.extract y'

prop_trivial_is_serializable :: Property
prop_trivial_is_serializable = withTests 1 $ property $ do
  let layer = Trivial
      bs = encode layer
      dec = decode bs :: Either String Trivial
  assert $ isRight dec

prop_can_randomise_trivial :: Property
prop_can_randomise_trivial = withTests 1 $ property $ do
  gen <- evalIO create
  _ :: Trivial <- evalIO $ createRandomWith UniformInit gen
  success

prop_can_show_trivial :: Property
prop_can_show_trivial = withTests 1 $ property $ do
  show Trivial `seq` success

prop_can_update_trivial_and_use_in_batches :: Property
prop_can_update_trivial_and_use_in_batches = property $ do
  let layer = Trivial
  runUpdate defSGD layer () `seq` success
  runUpdate defAdam layer () `seq` success
  reduceGradient @Trivial [()] `seq` success

tests :: IO Bool
tests = checkParallel $$(discover)
