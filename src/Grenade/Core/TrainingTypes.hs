{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE DuplicateRecordFields     #-}
{-# LANGUAGE BangPatterns              #-}

module Grenade.Core.TrainingTypes where

import           Data.Singletons.Prelude

import           Grenade.Core.Shape
import           Grenade.Core.Network
import           Grenade.Core.Optimizer

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
  deriving Show

data TrainingOptions 
  = forall opt. TrainingOptions { optimizer      :: Optimizer opt
                                , batchSize      :: Int
                                , validationFreq :: Int
                                , verbose        :: VerboseOptions
                                , metrics        :: [LossMetric]
                                }

data LossFunction shape = LossFunction (shape -> shape -> shape)
