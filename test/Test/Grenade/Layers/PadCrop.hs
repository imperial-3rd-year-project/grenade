{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

module Test.Grenade.Layers.PadCrop where

import           Grenade

import           Hedgehog

import           Numeric.LinearAlgebra.Static ( norm_Inf )

import           Test.Hedgehog.Hmatrix

import qualified Data.ByteString              as BS
import           Data.Serialize
import           Data.Either
import           System.Random.MWC             (create)

type PadCropNet3D = Network '[Pad 2 3 4 6, Crop 2 3 4 6] '[ 'D3 7 9 5, 'D3 16 15 5, 'D3 7 9 5 ]
type PadCropNet2D = Network '[Pad 2 3 4 6, Crop 2 3 4 6] '[ 'D2 7 9, 'D2 16 15, 'D2 7 9 ]

prop_pad_crop :: Property
prop_pad_crop =
  let net :: PadCropNet3D
      net = Pad :~> Crop :~> NNil
  in  property $
    forAll genOfShape >>= \(d :: S ('D3 7 9 5)) ->
      let (tapes, res)  = runForwards  net d
          (_    , grad) = runBackwards net tapes d
      in  do assert $ d ~~~ res
             assert $ grad ~~~ d

prop_pad_crop_2d :: Property
prop_pad_crop_2d =
  let net :: Network '[Pad 2 3 4 6, Crop 2 3 4 6] '[ 'D2 7 9, 'D2 16 15, 'D2 7 9 ]
      net = Pad :~> Crop :~> NNil
  in  property $
    forAll genOfShape >>= \(d :: S ('D2 7 9)) ->
      let (tapes, res)  = runForwards  net d
          (_    , grad) = runBackwards net tapes d
      in  do assert $ d ~~~ res
             assert $ grad ~~~ d

prop_pad_crop_is_serializable :: Property
prop_pad_crop_is_serializable = withTests 1 $ property $ do
  gen <- evalIO create
  pad  :: Pad 2 3 4 6  <- evalIO $ createRandomWith UniformInit gen
  crop :: Crop 2 3 4 6 <- evalIO $ createRandomWith UniformInit gen
  let net :: PadCropNet3D
      net = pad :~> crop :~> NNil
      bs = encode net
      dec = decode bs :: Either String PadCropNet3D
  assert $ isRight dec

prop_can_show_pad_crop :: Property
prop_can_show_pad_crop = 
  let net :: PadCropNet3D
      net = Pad :~> Crop :~> NNil
  in (withTests 1 . property) $ show net `seq` success

prop_can_update_pad_crop_and_use_in_batches :: Property
prop_can_update_pad_crop_and_use_in_batches =
  let net :: PadCropNet3D
      net = Pad :~> Crop :~> NNil
  in  property $
    forAll genOfShape >>= \(d :: S ('D3 7 9 5)) ->
      let (tapes, _) = runForwards  net d
          (v    , _)   = runBackwards net tapes d
      in do
        runUpdate defSGD net v `seq` success
        runUpdate defAdam net v `seq` success
        reduceGradient @PadCropNet3D [v] `seq` success
        

(~~~) :: S x -> S x -> Bool
(S1D x) ~~~ (S1D y) = norm_Inf (x - y) < 0.00001
(S2D x) ~~~ (S2D y) = norm_Inf (x - y) < 0.00001
(S3D x) ~~~ (S3D y) = norm_Inf (x - y) < 0.00001


tests :: IO Bool
tests = checkParallel $$(discover)
