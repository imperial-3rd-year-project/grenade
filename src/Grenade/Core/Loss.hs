{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE GADTs            #-}

{--
Module to define the loss functions and their derivatives. 
While only the derivatives are needed for backpropagation,
the loss functions themselves may be useful for measuring 
progress.
--}

module Grenade.Core.Loss (
    LossMetric (..)

  , quadratic
  , quadratic'
  , crossEntropy
  , crossEntropy'
  , exponential
  , exponential'
  , hellinger
  , hellinger'
  , kullbackLeibler
  , kullbackLeibler'
  , genKullbackLeibler
  , genKullbackLeibler'
  , itakuraSaito
  , itakuraSaito'
  ) where

import Grenade.Core.Shape
import Grenade.Core.TrainingTypes
import Data.Singletons ( SingI )
import Numeric.LinearAlgebra.HMatrix
import Numeric.LinearAlgebra.Static

-- | Helper function that sums the elements of a matrix
nsum :: SingI s => S s -> Double
nsum (S1D x) = sumElements $ extract x
nsum (S2D x) = sumElements $ extract x
nsum (S3D x) = sumElements $ extract x

quadratic :: SingI s => S s -> S s -> Double
quadratic x y = 0.5 * nsum ((x - y) ^ 2)

quadratic' :: SingI s => LossFunction (S s) 
quadratic' = LossFunction (-)

crossEntropy :: SingI s => S s -> S s -> Double
crossEntropy x y = - (nsum $ y * (log x) + ((nk 1) - y) * (log ((nk 1) - x)))

crossEntropy' :: SingI s => LossFunction (S s) 
crossEntropy' = LossFunction $ \x y -> (x - y) / ( ((fromInteger 1) - x) * x )

exponential :: SingI s => Double -> S s -> S s -> Double
exponential t x y = t * (exp (1/t * total))
  where
    total = nsum ((x - y) ^ 2)

exponential' :: SingI s => Double -> LossFunction (S s)
exponential' t = LossFunction $ \x y -> (nk 2)/(nk t) * (x - y) * (nk $ exponential t x y)

hellinger :: SingI s => S s -> S s -> Double
hellinger x y = 1/(sqrt 2) * (nsum $ (sqrt x - sqrt y) ^ 2)

hellinger' :: SingI s => LossFunction (S s)
hellinger' = LossFunction $ \x y -> ((sqrt x) - (sqrt y)) / ((nk $ sqrt 2) * sqrt x)

kullbackLeibler :: SingI s => S s -> S s -> Double
kullbackLeibler x y = nsum (y * log (y / x))

kullbackLeibler' :: (SingI s) => LossFunction (S s)
kullbackLeibler' = LossFunction $ \x y -> -(y / x)

genKullbackLeibler :: SingI s => S s -> S s -> Double
genKullbackLeibler x y = (kullbackLeibler x y) - (nsum y) + (nsum x)

genKullbackLeibler' :: (SingI s) => LossFunction (S s)
genKullbackLeibler' = LossFunction $ \x y -> (x - y)/x

itakuraSaito :: SingI s => S s -> S s -> Double
itakuraSaito x y = nsum ((y/x) - (log (y/x)) - (nk 1))

itakuraSaito' :: (SingI s) => LossFunction (S s)
itakuraSaito' = LossFunction $ \x y -> (x - y)/(x * x)





