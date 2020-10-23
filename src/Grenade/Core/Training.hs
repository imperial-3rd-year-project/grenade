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

fit :: (CreatableNetwork layers shapes
       , SingI (Last shapes)
       , NFData (Network layers shapes))

       => [(S (Head shapes), S (Last shapes))]
       -> [(S (Head shapes), S (Last shapes))]
       -> TrainingOptions
       -> Int
       -> IO (Network layers shapes)
fit trainRows validateRows TrainingOptions{ optimizer = opt, loss = l} epochs = do
    -- initialise the network with random weights
    net0 <- randomNetwork
    -- then training it over the epochs
    let !trainData = trainingData trainRows
        !valData   = trainingData validateRows
    foldM (runEpoch opt trainData valData) net0 [1..epochs]

runEpoch :: SingI (Last shapes)
         => Optimizer opt
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> TrainingData (S (Head shapes)) (S (Last shapes))
         -> Network layers shapes
         -> Int
         -> IO (Network layers shapes)
runEpoch opt (TrainingData t ts) (TrainingData v vs) net epoch = do 
  putStrLn $ "Training epoch " ++ show epoch
  pb <- newProgressBar defStyle 10 (Progress 0 t ())

  -- actual training
  (!trained, loss) <- foldM (combineTraining pb (sgdUpdateLearningParamters opt)) (net, 0) ts
  
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
                     -> (Network layers shapes, RealNum) 
                     -> (S (Head shapes), S (Last shapes))
                     -> IO (Network layers shapes, RealNum)
combineTraining pb !opt (!net, loss) (!x, !y) 
  = let (!net', loss') = train' opt net x y (LossFunction (-))
    in incProgress pb 1 >> return (net', loss + loss')

-- combine_val :: Network layers shapes -> S (Last shapes) -> (S (Head shapes), S (Last shapes)) -> S (Last shapes)
combine_val net loss (!x, !y) 
  = undefined --let !loss' = validate net x y (LossFunction (-))
    --in loss