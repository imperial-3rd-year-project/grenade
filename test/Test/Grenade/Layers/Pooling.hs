{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Pooling where

import           Data.Proxy
import           Data.Singletons ()

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           GHC.TypeLits
import           Grenade.Layers.Pooling
import           Grenade.Core

import           Hedgehog

import           Test.Hedgehog.Compat

import           Data.Serialize
import           Data.Either
import           System.Random.MWC             (create)

data OpaquePooling :: Type where
     OpaquePooling :: (KnownNat kh, KnownNat kw, KnownNat sh, KnownNat sw) => Pooling kh kw sh sw -> OpaquePooling

instance Show OpaquePooling where
    show (OpaquePooling n) = show n

genOpaquePooling :: Gen OpaquePooling
genOpaquePooling = do
    ~(Just kernelHeight) <- someNatVal <$> choose 2 15
    ~(Just kernelWidth ) <- someNatVal <$> choose 2 15
    ~(Just strideHeight) <- someNatVal <$> choose 2 15
    ~(Just strideWidth ) <- someNatVal <$> choose 2 15

    case (kernelHeight, kernelWidth, strideHeight, strideWidth) of
       (SomeNat (_ :: Proxy kh), SomeNat (_ :: Proxy kw), SomeNat (_ :: Proxy sh), SomeNat (_ :: Proxy sw)) ->
            return $ OpaquePooling (Pooling :: Pooling kh kw sh sw)

prop_pool_layer_witness =
  property $ do
    onet <- forAll genOpaquePooling
    case onet of
      (OpaquePooling (Pooling :: Pooling kernelRows kernelCols strideRows strideCols)) ->
        assert True


prop_pool_is_serializable :: Property
prop_pool_is_serializable = withTests 1 $ property $ do
  OpaquePooling (pool :: Pooling a b c d) <- blindForAll genOpaquePooling
  let bs = encode pool
      dec = decode bs :: Either String (Pooling a b c d)
  assert $ isRight dec

prop_can_randomise_pool :: Property
prop_can_randomise_pool = withTests 1 $ property $ do
  gen <- evalIO create
  _ :: Pooling 5 12 8 3 <- evalIO $ createRandomWith UniformInit gen
  success

prop_can_show_pool :: Property
prop_can_show_pool = withTests 1 $ property $ do
  OpaquePooling pool <- blindForAll genOpaquePooling
  show pool `seq` success

prop_can_update_pool_and_use_in_batches :: Property
prop_can_update_pool_and_use_in_batches = property $ do
  OpaquePooling (pool :: Pooling a b c d) <- blindForAll genOpaquePooling
  runUpdate defSGD pool () `seq` success
  runUpdate defAdam pool () `seq` success
  reduceGradient @(Pooling a b c d) [()] `seq` success

tests :: IO Bool
tests = checkParallel $$(discover)
