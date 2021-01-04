{-# LANGUAGE AllowAmbiguousTypes       #-}
{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DuplicateRecordFields     #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE NamedFieldPuns            #-}
{-|
Module      : Grenade.Core.Training
Description : Training functions for neural networks
Copyright   : (c) Theo Charalambous, 2020
License     : BSD2
Stability   : experimental

This module defines some training functionality for supervised neural networks.
-}
module Grenade.Core.Training
  (
  -- * Training functions
    VerboseOption(..)
  , TrainingOptions(..)
  , fit

  -- * Default training settings
  , defaultSGDOptions
  , defaultAdamOptions
  ) where

import           Control.Monad

import           Data.List                  (foldl')
import           Data.List.Split            (chunksOf)
import           Data.Singletons.Prelude

import           Grenade.Core.Loss
import           Grenade.Core.Network
import           Grenade.Core.Optimizer
import           Grenade.Core.Runner
import           Grenade.Core.Shape
import           Grenade.Types

import           Numeric.Limits             (infinity)

import           System.ProgressBar

data VerboseOption = Silent   -- ^ no logging to console during training
                   | Minimal  -- ^ logs the validation score at the end of each epoch
                   | Full     -- ^ training bar to indicate progress in each epoch
                              --   with validation score at the end
  deriving (Eq, Show)

-- | sized list of tuples for supervised neural network training, prevents having to recalculate
--   the size of the training/validation data for each epoch
data TrainingData inputShape outputShape = TrainingData Int [(inputShape, outputShape)]

-- | smart constructor for TrainingData, calcualtes the length of the list
trainingData :: [(S inputShape, S outputShape)] -> TrainingData (S inputShape) (S outputShape)
trainingData xs = let !n = length xs
                  in  TrainingData n xs

data TrainingOptions
  = forall opt. TrainingOptions { optimizer      :: Optimizer opt  -- ^ Optimizer to perform gradient descent with
                                , batchSize      :: Int            -- ^ number of training examples run before a parameter update
                                , validationFreq :: Int            -- ^ number of epochs between each validation on the held out validation set
                                , verbose        :: VerboseOption
                                , metrics        :: [LossMetric]   -- ^ loss functions
                                }

-- | Using the default options of 'defSGD', with a batch size of 1, validation frequency of 1,
--   full verbosity and quadratic loss for backpropogation.
defaultSGDOptions :: TrainingOptions
defaultSGDOptions = TrainingOptions { optimizer      = defSGD
                                    , batchSize      = 1
                                    , validationFreq = 1
                                    , verbose        = Full
                                    , metrics        = [Quadratic]
                                    }

-- | Using the default options of 'defAdam', with a batch size of 1, validation frequency of 1,
--   full verbosity and quadratic loss for backpropogation.
defaultAdamOptions :: TrainingOptions
defaultAdamOptions = TrainingOptions { optimizer      = defAdam
                                     , batchSize      = 1
                                     , validationFreq = 1
                                     , verbose        = Full
                                     , metrics        = [Quadratic]
                                     }

-- | Given training data and validation data, fits a network to that dataset
fit :: (CreatableNetwork layers shapes
       , SingI (Last shapes))
       => [(S (Head shapes), S (Last shapes))]  -- ^ training data
       -> [(S (Head shapes), S (Last shapes))]  -- ^ validation data
       -> TrainingOptions                       -- ^ training options
       -> Int                                   -- ^ number of epochs
       -> LossFunction (S (Last shapes))        -- ^ loss function for backpropogation
       -> IO (Network layers shapes)            -- ^ the trained network
fit trainRows validateRows TrainingOptions{ optimizer, batchSize, validationFreq, verbose, metrics } epochs lossFnc = do
    let !trainData = trainingData trainRows
        !valData   = trainingData validateRows
    net0        <- randomNetwork
    (_, net, _) <- foldM (runEpoch optimizer trainData valData lossFnc metrics batchSize verbose validationFreq) (net0, net0, infinity)  [1..epochs]
    return net

runEpoch :: SingI (Last shapes)
         => Optimizer opt
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> LossFunction (S (Last shapes))
         -> [LossMetric]
         -> Int
         -> VerboseOption
         -> Int
         -> (Network layers shapes, Network layers shapes, RealNum)
         -> Int
         -> IO (Network layers shapes, Network layers shapes, RealNum)
runEpoch opt (TrainingData t ts) valData lossFnc ms batchSize verbosity validationFreq (net, minNet, minLoss) epoch = do
  when (verbosity /= Silent) $ putStrLn $ "Training epoch " ++ show epoch

  pb <- if verbosity == Full
        then Just <$> newProgressBar defStyle 10 (Progress 0 t ())
        else return Nothing

  -- training
  (!trainedNet, loss) <- if batchSize == 1
                         then let updates = sgdUpdateLearningParameters opt
                               in foldM (combineTraining pb updates lossFnc) (net, 0) ts
                         else let bs      = chunksOf batchSize ts
                                  updates = sgdUpdateLearningParameters opt
                                  f       = combineBatchTraining pb updates lossFnc batchSize
                               in foldM f (net, 0) bs

  -- Validating data and printing losses.
  when (mod epoch validationFreq == 0 && verbosity /= Silent) $ do
    putStrLn $ "Loss:     " ++ show (loss / fromIntegral t)
    mapM_ (\m -> putStrLn $ "Val " ++ show m ++ ": " ++ show (validate' trainedNet valData m)) ms
    putStr "\n"

  let (newMinNet, newMinLoss) = if loss <= minLoss then (trainedNet, loss) else (minNet, minLoss)
  return (trainedNet, newMinNet, newMinLoss)

sgdUpdateLearningParameters :: Optimizer opt -> Optimizer opt
sgdUpdateLearningParameters (OptSGD rate mom reg) = OptSGD rate mom (reg * 10)
sgdUpdateLearningParameters o                     = o

combineTraining :: Maybe (ProgressBar ())
                -> Optimizer opt
                -> LossFunction (S (Last shapes))
                -> (Network layers shapes, RealNum)
                -> (S (Head shapes), S (Last shapes))
                -> IO (Network layers shapes, RealNum)
combineTraining pb !opt lossFnc (!net, loss) (!x, !y)
  = let (!net', loss') = train opt net x y lossFnc
    in  case pb of 
      Nothing  -> return (net', loss + loss')
      Just pb' -> incProgress pb' 1 >> return (net', loss + loss')

combineBatchTraining :: Maybe (ProgressBar ())
                     -> Optimizer opt
                     -> LossFunction (S (Last shapes))
                     -> Int
                     -> (Network layers shapes, RealNum)
                     -> [(S (Head shapes), S (Last shapes))]
                     -> IO (Network layers shapes, RealNum)
combineBatchTraining pb !opt lossFnc batchSize (!net, loss) ts
  = let (xs, ys)       = unzip ts
        (!net', loss') = batchTrain opt net xs ys lossFnc
    in case pb of 
      Nothing  -> return (net', loss + loss') 
      Just pb' -> incProgress pb' batchSize >> return (net', loss + loss')

validate' :: SingI (Last shapes)
          => Network layers shapes -> TrainingData (S (Head shapes)) (S (Last shapes)) -> LossMetric -> RealNum
validate' net (TrainingData v vs) metric
  = case metric of
      Quadratic               -> validateWithLoss quadratic
      CrossEntropy            -> validateWithLoss crossEntropy
      Exponential             -> validateWithLoss (exponential 1)
      Hellinger               -> validateWithLoss hellinger
      KullbackLeibler         -> validateWithLoss kullbackLeibler
      GenKullbackLeibler      -> validateWithLoss genKullbackLeibler
      ItakuraSaito            -> validateWithLoss itakuraSaito
      CategoricalCrossEntropy -> validateWithLoss categoricalCrossEntropy
  where
    validateWithLoss l = (/ fromIntegral v) $ foldl' (\n (x, y) -> n + l (runNet net x) y) 0 vs
