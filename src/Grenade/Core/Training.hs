{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE DuplicateRecordFields     #-}
{-# LANGUAGE AllowAmbiguousTypes       #-}

module Grenade.Core.Training where

import           Control.DeepSeq
import           Control.Monad

import           Data.List                   (foldl')
import           Data.List.Split             (chunksOf)
import           Data.Singletons.Prelude

import qualified Numeric.LinearAlgebra.Static as SA

import           Grenade.Core.Loss
import           Grenade.Core.Network
import           Grenade.Core.Optimizer
import           Grenade.Core.Shape
import           Grenade.Core.Runner
import           Grenade.Core.TrainingTypes
import           Grenade.Types
import           Grenade.Utils.LinearAlgebra

import           System.ProgressBar

fit :: (CreatableBatchNetwork layers shapes
       , SingI (Last shapes)
       , NFData (Network layers shapes))
       => [(S (Head shapes), S (Last shapes))]
       -> [(S (Head shapes), S (Last shapes))]
       -> TrainingOptions
       -> Int
       -> LossFunction (S (Last shapes))
       -> IO (BatchNetwork layers shapes)
fit trainRows validateRows TrainingOptions{ optimizer=opt, batchSize=bs, metrics=ms } epochs lossFnc = do
    -- initialise the network with random weights
    net0 <- randomBatchNetwork
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
         -> BatchNetwork layers shapes
         -> Int
         -> IO (BatchNetwork layers shapes)
runEpoch opt (TrainingData t ts) valData lossFnc ms batchSize net epoch = do 
  putStrLn $ "Training epoch " ++ show epoch
  pb <- newProgressBar defStyle 10 (Progress 0 t ())

  -- training
  (!trained, loss) <- let batchedData = chunksOf batchSize ts 
                      in foldM (combineBatchTraining pb (sgdUpdateLearningParamters opt) lossFnc batchSize) (net, 0) batchedData

  -- validate trained model
  let !val_losses  = map (\m -> (m, validate' trained valData m)) ms

  putStrLn $ "Loss:     " ++ show (loss / fromIntegral t)
  -- print the validation losses
  mapM_ (\(m, x) -> putStrLn $ "Val " ++ show m ++ ": " ++ show x) val_losses
  putStrLn ""
  return trained

sgdUpdateLearningParamters :: Optimizer opt -> Optimizer opt
sgdUpdateLearningParamters (OptSGD rate mom reg) = OptSGD rate mom (reg * 10)
sgdUpdateLearningParamters o                     = o

combineTraining :: SingI (Last shapes)
                => ProgressBar ()
                -> Optimizer opt 
                -> LossFunction (S (Last shapes))
                -> (Network layers shapes, RealNum) 
                -> (S (Head shapes), S (Last shapes))
                -> IO (Network layers shapes, RealNum)
combineTraining pb !opt lossFnc (!net, loss) (!x, !y) 
  = let (!net', loss') = train opt net x y lossFnc
    in incProgress pb 1 >> return (net', loss + loss')

combineBatchTraining :: SingI (Last shapes)
                     => ProgressBar ()
                     -> Optimizer opt 
                     -> LossFunction (S (Last shapes))
                     -> Int
                     -> (BatchNetwork layers shapes, RealNum) 
                     -> [(S (Head shapes), S (Last shapes))]
                     -> IO (BatchNetwork layers shapes, RealNum)
combineBatchTraining pb !opt lossFnc batchSize (!net, loss) ts 
  = let (xs, ys)       = unzip ts
        (!net', loss') = batchTrain opt net xs ys lossFnc
    in incProgress pb batchSize >> return (net', loss + loss')


validate' :: SingI (Last shapes) 
          => BatchNetwork layers shapes -> TrainingData (S (Head shapes)) (S (Last shapes)) -> LossMetric -> RealNum
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
    validateWithLoss l = (/ fromIntegral v) $ foldl' (\n (x, y) -> n + l (runBatchNet net x) y) 0 vs