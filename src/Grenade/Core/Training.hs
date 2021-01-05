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

import           Numeric.Limits             (infinity)

import           System.ProgressBar

{-|
Module      : Grenade.Core.Training
Description : Contains method used for network training 

-}


-- | Fits the network to model the training data as good as possible
fit :: (CreatableNetwork layers shapes
       , SingI (Last shapes))
       => [(S (Head shapes), S (Last shapes))]
       -> [(S (Head shapes), S (Last shapes))]
       -> TrainingOptions
       -> Int
       -> LossFunction (S (Last shapes))
       -> IO (Network layers shapes)
fit trainRows validateRows TrainingOptions{ optimizer=opt, batchSize=bs, metrics=ms } epochs lossFnc = do
    let !trainData = trainingData trainRows
        !valData   = trainingData validateRows
    net0        <- randomNetwork
    (_, net, _) <- foldM (runEpoch opt trainData valData lossFnc ms bs) (net0, net0, infinity)  [1..epochs]
    return net

-- | TODO Theo
runEpoch :: SingI (Last shapes)
         => Optimizer opt
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> LossFunction (S (Last shapes))
         -> [LossMetric]
         -> Int
         -> (Network layers shapes, Network layers shapes, RealNum)
         -> Int
         -> IO (Network layers shapes, Network layers shapes, RealNum)
runEpoch opt (TrainingData t ts) valData lossFnc ms batchSize (net, minNet, minLoss) epoch = do
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

  -- Validating data and printing losses.
  putStrLn $ "Loss:     " ++ show (loss / fromIntegral t)
  mapM_ (\m -> putStrLn $ "Val " ++ show m ++ ": " ++ show (validate' trained valData m)) ms
  putStrLn ""

  let (newMinNet, newMinLoss) = if loss <= minLoss then (trained, loss) else (minNet, minLoss)
  return (trained, newMinNet, newMinLoss)

-- | TODO Theo
sgdUpdateLearningParameters :: Optimizer opt -> Optimizer opt
sgdUpdateLearningParameters (OptSGD rate mom reg) = OptSGD rate mom (reg * 10)
sgdUpdateLearningParameters o                     = o

-- | TODO Theo
combineTraining :: ProgressBar ()
                -> Optimizer opt
                -> LossFunction (S (Last shapes))
                -> (Network layers shapes, RealNum)
                -> (S (Head shapes), S (Last shapes))
                -> IO (Network layers shapes, RealNum)
combineTraining pb !opt lossFnc (!net, loss) (!x, !y)
  = let (!net', loss') = train opt net x y lossFnc
    in incProgress pb 1 >> return (net', loss + loss')

-- | TODO Theo
combineBatchTraining :: ProgressBar ()
                     -> Optimizer opt
                     -> LossFunction (S (Last shapes))
                     -> Int
                     -> (Network layers shapes, RealNum)
                     -> [(S (Head shapes), S (Last shapes))]
                     -> IO (Network layers shapes, RealNum)
combineBatchTraining pb !opt lossFnc batchSize (!net, loss) ts
  = let (xs, ys)       = unzip ts
        (!net', loss') = batchTrain opt net xs ys lossFnc
    in incProgress pb batchSize >> return (net', loss + loss')


-- | Calculates the loss with respect to the given loss metric
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
