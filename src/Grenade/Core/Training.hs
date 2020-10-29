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
fit trainRows validateRows TrainingOptions{ optimizer=opt, loss=l, batchSize=bs } epochs lossFnc = do
    -- initialise the network with random weights
    net0 <- randomBatchNetwork
    -- then training it over the epochs
    let !trainData = trainingData trainRows
        !valData   = trainingData validateRows
    foldM (runEpoch opt trainData valData lossFnc bs) net0 [1..epochs]

runEpoch :: SingI (Last shapes)
         => Optimizer opt
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> LossFunction (S (Last shapes))
         -> Int
         -> BatchNetwork layers shapes
         -> Int
         -> IO (BatchNetwork layers shapes)
runEpoch opt (TrainingData t ts) (TrainingData v vs) lossFnc batchSize net epoch = do 
  putStrLn $ "Training epoch " ++ show epoch
  pb <- newProgressBar defStyle 10 (Progress 0 t ())

  (!trained, loss) <- let batchedData = chunksOf batchSize ts 
                      in foldM (combineBatchTraining pb (sgdUpdateLearningParamters opt) lossFnc batchSize) (net, 0) batchedData

  -- actual training
  -- (!trained, loss) <- foldM (combineTraining pb (sgdUpdateLearningParamters opt) lossFnc) (net, 0) ts
  
  -- validate trained model
  -- vs :: [(S (Head shapes), S (Last shapes))]
  -- let !val_loss     = foldl' (combine_val trained) 0 vs

  putStrLn $ "Loss:     " ++ show (loss / fromIntegral t)
  --putStrLn $ "Val Loss: " ++ show ((sumV $ val_loss) / fromIntegral v)
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


-- combine_val :: Network layers shapes -> S (Last shapes) -> (S (Head shapes), S (Last shapes)) -> S (Last shapes)
combine_val net loss (!x, !y) 
  = undefined --let !loss' = validate net x y (LossFunction (-))
    --in loss