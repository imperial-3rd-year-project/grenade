{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}

module Test.Grenade.Layers.Reshape where

import           Data.Proxy
import           Data.Singletons ()

import           GHC.TypeLits
import           Grenade.Layers.Reshape
import           Grenade.Core

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

import           Data.Serialize
import           Data.Either
import           System.Random.MWC             (create)

prop_can_reshape_2d_to_1d :: Property
prop_can_reshape_2d_to_1d = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w))) -> do
      input :: S ('D2 h w) <- forAll genOfShape
      let (t, y :: S ('D1 (h * w))) = runForwards  Reshape input
          (_, w :: S ('D2 h w))     = runBackwards Reshape t y
          out = extractVec y
          inp = concat $ extractMat input
      (extractMat w) === (extractMat input)
      out === inp

prop_can_reshape_3d_to_1d :: Property
prop_can_reshape_3d_to_1d = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100
  channels :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c))) -> do
      input :: S ('D3 h w c) <- forAll genOfShape
      let (t, y :: S ('D1 (h * w * c))) = runForwards  Reshape input
          (_, w :: S ('D3 h w c))       = runBackwards Reshape t y
          out = extractVec y
          inp = concat $ extractMat3D input
      (extractMat3D w) === (extractMat3D input)
      out === inp

prop_can_reshape_3d_to_2d :: Property
prop_can_reshape_3d_to_2d = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w))) -> do
      input :: S ('D3 h w 1) <- forAll genOfShape
      let (t, y :: S ('D2 h w))   = runForwards  Reshape input
          (_, w :: S ('D3 h w 1)) = runBackwards Reshape t y
          out = extractMat y
          inp = extractMat3D input
      (extractMat3D w) === inp
      out === inp

prop_can_reshape_2d_to_3d :: Property
prop_can_reshape_2d_to_3d = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w))) -> do
      input :: S ('D2 h w) <- forAll genOfShape
      let (t, y :: S ('D3 h w 1)) = runForwards  Reshape input
          (_, w :: S ('D2 h w))   = runBackwards Reshape t y
          out = extractMat3D y
          inp = extractMat input
      (extractMat w) === inp
      out === inp

prop_can_reshape_1d_to_2d :: Property
prop_can_reshape_1d_to_2d = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w))) -> do
      input :: S ('D1 (h * w)) <- forAll genOfShape
      let (t, y :: S ('D2 h w))     = runForwards  Reshape input
          (_, w :: S ('D1 (h * w))) = runBackwards Reshape t y
          out = concat $ extractMat y
          inp = extractVec input
      (extractVec w) === inp
      out === inp

prop_can_reshape_1d_to_3d :: Property
prop_can_reshape_1d_to_3d = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100
  channels :: Int <- forAll $ choose 2 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels)) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c))) -> do
      input :: S ('D1 (h * w * c)) <- forAll genOfShape
      let (t, y :: S ('D3 h w c))       = runForwards  Reshape input
          (_, w :: S ('D1 (h * w * c))) = runBackwards Reshape t y
          out = concat $ extractMat3D y
          inp = extractVec input
      (extractVec w) === inp
      out === inp

prop_can_reshape_3d_to_4d :: Property
prop_can_reshape_3d_to_4d = withTests 1 $ property $ do
  height      :: Int <- forAll $ choose 2 20
  width       :: Int <- forAll $ choose 2 20
  channels    :: Int <- forAll $ choose 2 20
  temp        :: Int <- forAll $ choose 2 20
  let tup = (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels), someNatVal (fromIntegral temp))
  case tup of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c)), Just (SomeNat (Proxy :: Proxy t))) -> do
      input :: S ('D3 (h * w) c t) <- forAll genOfShape
      let (t, y :: S ('D4 h w c t))     = runForwards  Reshape input
          (_, w :: S ('D3 (h * w) c t)) = runBackwards Reshape t y
          out = concat $ extractMat4D y
          inp = concat $ extractMat3D input
      (concat $ extractMat3D w) === inp
      out === inp

prop_can_reshape_4d_to_3d :: Property
prop_can_reshape_4d_to_3d = withTests 1 $ property $ do
  height      :: Int <- forAll $ choose 2 20
  width       :: Int <- forAll $ choose 2 20
  channels    :: Int <- forAll $ choose 2 20
  temp        :: Int <- forAll $ choose 2 20
  let tup = (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels), someNatVal (fromIntegral temp))
  case tup of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c)), Just (SomeNat (Proxy :: Proxy t))) -> do
      input :: S ('D4 h w c t) <- forAll genOfShape
      let (t, y :: S ('D3 (h * w) c t)) = runForwards  Reshape input
          (_, w :: S ('D4 h w c t))     = runBackwards Reshape t y
          out = concat $ extractMat3D y
          inp = concat $ extractMat4D input
      (concat $ extractMat4D w) === inp
      out === inp

prop_reshape_is_serializable :: Property
prop_reshape_is_serializable = withTests 1 $ property $ do
  let res = Reshape
      bs = encode res
      dec = decode bs :: Either String Reshape
  assert $ isRight dec

prop_can_randomise_reshape :: Property
prop_can_randomise_reshape = withTests 1 $ property $ do
  gen <- evalIO create
  (_ :: Reshape) <- evalIO $ createRandomWith UniformInit gen
  success

prop_can_show_reshape :: Property
prop_can_show_reshape = withTests 1 $ property $ do
  show Reshape `seq` success

prop_can_update_reshape_and_use_in_batches :: Property
prop_can_update_reshape_and_use_in_batches = property $ do
  let res = Reshape
  runUpdate defSGD res () `seq` success
  runUpdate defAdam res () `seq` success
  reduceGradient @Reshape [()] `seq` success

tests :: IO Bool
tests = checkParallel $$(discover)
