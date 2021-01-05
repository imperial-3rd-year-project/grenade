{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE DuplicateRecordFields     #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-|
Module      : Grenade.Core.TrainingTypes
Description : Defines the types and default options for NN training

-}

module Grenade.Core.TrainingTypes where

import           Grenade.Core.Optimizer
import           Grenade.Core.Shape

data VerboseOptions = Silent | Minimal | Full

data TrainingData inputShape outputShape = TrainingData Int [(inputShape, outputShape)]

trainingData :: [(S inputShape, S outputShape)] -> TrainingData (S inputShape) (S outputShape)
trainingData xs = let !n = length xs
                  in  TrainingData n xs

data LossMetric = Quadratic
                | CrossEntropy
                | Exponential
                | Hellinger
                | KullbackLeibler
                | GenKullbackLeibler
                | ItakuraSaito
                | CategoricalCrossEntropy
  deriving Show

data TrainingOptions
  = forall opt. TrainingOptions { optimizer      :: Optimizer opt
                                , batchSize      :: Int
                                , validationFreq :: Int
                                , verbose        :: VerboseOptions
                                , metrics        :: [LossMetric]
                                }

defaultSGDOptions :: TrainingOptions
defaultSGDOptions = TrainingOptions { optimizer      = OptSGD 0.01 0.9 0.0005
                                    , batchSize      = 1
                                    , validationFreq = 1
                                    , verbose        = Full
                                    , metrics        = [Quadratic]
                                    }

defaultAdamOptions :: TrainingOptions
defaultAdamOptions = TrainingOptions { optimizer      = OptAdam 0.001 0.9 0.999 1e-4 1e-3
                                     , batchSize      = 1
                                     , validationFreq = 1
                                     , verbose        = Full
                                     , metrics        = [Quadratic]
                                     }

newtype LossFunction shape = LossFunction (shape -> shape -> shape)
