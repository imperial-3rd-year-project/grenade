{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Nonlinear where

import           Data.Singletons

import           Grenade

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix
import           Test.Hedgehog.TypeLits

import           Numeric.LinearAlgebra.Static ( norm_Inf )

prop_sigmoid_grad :: Property
prop_sigmoid_grad = property $
    forAllWith rss genShape >>= \case
        (SomeSing (r :: Sing s)) ->
            withSingI r $ 
                forAll genOfShape >>= \(ds :: S s) ->
                    let (tape, f  :: S s)  = runForwards Logit ds
                        ((), ret  :: S s)  = runBackwards Logit tape (1 :: S s)
                        (_, numer :: S s)  = runForwards Logit (ds + numericalGradDiff)
                        numericalGradient  = (numer - f) / numericalGradDiff
                    in isWithinOf precision 0 ((case numericalGradient - ret of
                         (S1D x) -> norm_Inf x
                         (S2D x) -> norm_Inf x
                         (S3D x) -> norm_Inf x) :: RealNum)

prop_tanh_grad :: Property
prop_tanh_grad = property $
    forAllWith rss genShape >>= \case
        (SomeSing (r :: Sing s)) ->
            withSingI r $
                blindForAll genOfShape >>=  \(ds :: S s) ->
                    let (tape, f  :: S s)  = runForwards Tanh ds
                        ((), ret  :: S s)  = runBackwards Tanh tape (1 :: S s)
                        (_, numer :: S s)  = runForwards Tanh (ds + numericalGradDiff)
                        numericalGradient  = (numer - f) / numericalGradDiff
                    in isWithinOf precision 0 ((case numericalGradient - ret of
                         (S1D x) -> norm_Inf x
                         (S2D x) -> norm_Inf x
                         (S3D x) -> norm_Inf x) :: RealNum)

tests :: IO Bool
tests = checkParallel $$(discover)
