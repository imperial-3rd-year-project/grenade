{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Shape where

import           Grenade.Core.Shape

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))
import           Numeric.LinearAlgebra.Static as H hiding ((===))
import           Numeric.LinearAlgebra.Devel as U 
import           Numeric.LinearAlgebra.Data as D hiding ((===))

import           Data.Maybe (fromJust)

import           Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import           GHC.TypeLits
import           Data.Proxy

import Debug.Trace

import           Test.Hedgehog.Compat

-- prop_batchMap_correctly_maps_1d_batches = property $ do
--   batches  <- forAll $ choose 2 100
--   width    <- forAll $ choose 2 100
--   xs       <- forAll $ Gen.list (Range.singleton (batches * width)) $ Gen.double (Range.linearFrac 0 100)
--   mul      <- forAll $ Gen.double (Range.linearFrac 0 100)

--   case (someNatVal $ toInteger batches, someNatVal $ toInteger width) of
--     (Just (SomeNat (Proxy :: Proxy b)), Just (SomeNat (Proxy :: Proxy w))) ->
--       let m  = H.matrix xs :: L b w
--           t  = T1D m
--           mr = H.matrix (map (* mul) xs) :: L b w
--           f :: S ('D1 w) -> S ('D1 w)
--             = (\(S1D v) -> S1D $ fromJust $ H.create $ mapVectorWithIndex (const (* mul)) $ H.extract v)
--       in (H.extract mr ===) $ (\(T1D m') -> H.extract m') $ batchMap f t

-- prop_batchMap_correctly_maps_2d_batches = property $ do
--   batches  <- forAll $ choose 2 100
--   width    <- forAll $ choose 2 100
--   height   <- forAll $ choose 2 100
--   xs       <- forAll $ Gen.list (Range.singleton (batches * width * height)) $ Gen.double (Range.linearFrac 1 100)
--   case (someNatVal $ toInteger batches, someNatVal $ toInteger width, someNatVal $ toInteger height) of
--     (Just (SomeNat (Proxy :: Proxy b)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy h))) ->
--       case (Proxy :: Proxy (KnownNat (w * h))) of 
--         _ ->
--           let v = D.vector xs
--               m = fromJust . H.create $ U.matrixFromVector U.RowMajor batches (width * height) v
--               t = T2D m :: T b ('D2 w h)
--           in (H.extract m ===) $ (\(T2D m') -> H.extract m') $ batchMap id t 

tests :: IO Bool
tests = checkParallel $$(discover)
