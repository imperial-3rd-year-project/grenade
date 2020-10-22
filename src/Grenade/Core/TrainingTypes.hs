{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE DuplicateRecordFields     #-}

module Grenade.Core.TrainingTypes where

import           Data.Singletons.Prelude

import           Grenade.Core.Shape 
import           Grenade.Core.Network
import           Grenade.Core.Optimizer

data VerboseOptions = Silent | Minimal | Full 

data TrainingData inputShape outputShape = TrainingData Int [(inputShape, outputShape)] 

trainingData :: [(S inputShape, S outputShape)] -> TrainingData (S inputShape) (S outputShape)
trainingData xs = TrainingData (length xs) xs

data TrainingOptions 
  = forall opt. TrainingOptions { optimizer      :: Optimizer opt
                                , batchSize      :: Int
                                , validationFreq :: Int
                                , loss           :: Metric
                                , verbose        :: VerboseOptions
                                , metrics        :: [Metric] 
                                }

-- data TrainingNetwork layers shapes
--   = TrainingNetwork { network   :: Network layers shapes 
--                     , epoch     :: Int 
--                     , maxExpoch :: Int 
--                     , loss      :: Metric
--                     }
-- 
-- data TrainedNetwork layers shapes
--   = TrainedNetwork { network   :: Network layers shapes 
--                    , loss      :: Loss (S (Last shapes))
--                    }

data LossFunction shape = LossFunction (shape -> shape -> shape) 

data Metric = Accuracy

type LossUnit = Double 

data Loss shape
  = Loss { lMetric :: Metric
         , lValue  :: shape
         }
