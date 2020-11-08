{-# LANGUAGE AllowAmbiguousTypes       #-}
{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE DuplicateRecordFields     #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}

module Grenade.Core.Training where

import           Control.Monad

import           Data.List                  (foldl')
import           Data.List.Split            (chunksOf)
import           Data.Singletons.Prelude

import           Grenade.Core.Loss
import           Grenade.Core.Network
import           Grenade.Core.Optimizer
import           Grenade.Core.Runner
import           Grenade.Core.Shape
import           Grenade.Core.TrainingTypes
import           Grenade.Types

import           System.ProgressBar

fit :: (CreatableNetwork layers shapes
       , SingI (Last shapes))
       => [(S (Head shapes), S (Last shapes))]
       -> [(S (Head shapes), S (Last shapes))]
       -> TrainingOptions
       -> Int
       -> LossFunction (S (Last shapes))
       -> IO (Network layers shapes)
fit trainRows validateRows TrainingOptions{ optimizer=opt, batchSize=bs, metrics=ms } epochs lossFnc = do
    -- initialise the network with random weights
    net0 <- randomNetwork
    -- then train it over the epochs
    let !trainData = trainingData trainRows
        !valData   = trainingData validateRows
    foldM (runEpoch opt trainData valData lossFnc ms bs) net0 [1..epochs]

runEpoch :: SingI (Last shapes)
         => Optimizer opt
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> LossFunction (S (Last shapes))
         -> [LossMetric]
         -> Int
         -> Network layers shapes
         -> Int
         -> IO (Network layers shapes)
runEpoch opt (TrainingData t ts) valData lossFnc ms batchSize net epoch = do
  putStrLn $ "Training epoch " ++ show epoch
  pb <- newProgressBar defStyle 10 (Progress 0 t ())

  -- training
  (!trained, loss) <- if batchSize == 1 
                         then let updates = (sgdUpdateLearningParameters opt)
                               in foldM (combineTraining pb updates lossFnc) (net, 0) ts
                         else let bs      = (chunksOf batchSize ts)
                                  updates = (sgdUpdateLearningParameters opt)
                                  f       = (combineBatchTraining pb updates lossFnc batchSize)
                               in foldM f (net, 0) bs

  -- validate trained model
  let !val_losses  = map (\m -> (m, validate' trained valData m)) ms

  putStrLn $ "Loss:     " ++ show (loss / fromIntegral t)
  -- print the validation losses
  mapM_ (\(m, x) -> putStrLn $ "Val " ++ show m ++ ": " ++ show x) val_losses
  putStrLn ""
  return trained

sgdUpdateLearningParameters :: Optimizer opt -> Optimizer opt
sgdUpdateLearningParameters (OptSGD rate mom reg) = OptSGD rate mom (reg * 10)
sgdUpdateLearningParameters o                     = o

combineTraining :: ProgressBar ()
                -> Optimizer opt
                -> LossFunction (S (Last shapes))
                -> (Network layers shapes, RealNum)
                -> (S (Head shapes), S (Last shapes))
                -> IO (Network layers shapes, RealNum)
combineTraining pb !opt lossFnc (!net, _) (!x, !y)
  = incProgress pb 1 >> return (train opt net x y lossFnc)

combineBatchTraining :: ProgressBar ()
                     -> Optimizer opt
                     -> LossFunction (S (Last shapes))
                     -> Int
                     -> (Network layers shapes, RealNum)
                     -> [(S (Head shapes), S (Last shapes))]
                     -> IO (Network layers shapes, RealNum)
combineBatchTraining pb !opt lossFnc batchSize (!net, loss) ts
  = incProgress pb batchSize >> return (batchTrain opt net xs ys lossFnc)
    where
      (xs, ys) = unzip ts



validate' :: SingI (Last shapes)
          => Network layers shapes -> TrainingData (S (Head shapes)) (S (Last shapes)) -> LossMetric -> RealNum
validate' net (TrainingData v vs) metric
  = case metric of
      Quadratic          -> validateWithLoss quadratic
      CrossEntropy       -> validateWithLoss crossEntropy
      Exponential        -> validateWithLoss (exponential 1)
      Hellinger          -> validateWithLoss hellinger
      KullbackLeibler    -> validateWithLoss kullbackLeibler
      GenKullbackLeibler -> validateWithLoss genKullbackLeibler
      ItakuraSaito       -> validateWithLoss itakuraSaito
  where
    validateWithLoss l = (/ fromIntegral v) $ foldl' (\n (x, y) -> n + l (runNet net x) y) 0 vs
