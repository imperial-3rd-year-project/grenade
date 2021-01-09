{-# LANGUAGE BangPatterns     #-}
{-# LANGUAGE CPP              #-}
{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-# LANGUAGE TemplateHaskell  #-}
{-# LANGUAGE TypeOperators    #-}

module Test.Grenade.Sys.Training where

-- TODO: These tests are unreliable and are not included.
-- In particular, there needs to be more investigation
-- into how to fix the seed that fit uses for random generation
-- or otherwise expose the combineTraining functions from the
-- Training module.

import qualified Numeric.LinearAlgebra.Data   as D
import qualified Numeric.LinearAlgebra.Static as H
import           GHC.TypeLits
import           Control.Monad.Random

import           Grenade

import           Test.Grenade.Sys.Utils

import           Hedgehog
import qualified Hedgehog.Gen                 as Gen
import qualified Hedgehog.Range               as Range

type FFNet 
  = Network 
    '[ FullyConnected 2 40, Tanh, 
       FullyConnected 40 10, Relu, 
       FullyConnected 10 1, Logit 
     ]
    '[ 'D1 2,          -- Input
       'D1 40, 'D1 40, -- Fully connected, Tanh
       'D1 10, 'D1 10, -- Fully connected, Relu
       'D1 1, 'D1 1    -- Fully connected, Logit
     ]

circle :: S ('D1 2) -> S ('D1 1)
circle (S1D v) =
  if v `inCircle` (fromRational 0, 0.5)
     then S1D $ fromRational 1
     else S1D $ fromRational 0
  where
    inCircle :: KnownNat n => H.R n -> (H.R n, RealNum) -> Bool
    u `inCircle` (o, r) = H.norm_2 (u - o) <= r

prop_training_decreases_loss :: Property
prop_training_decreases_loss = withTests 10 $ property $ do
    let n = 1000
    rand <- forAll $ Gen.int $ Range.linear 1 10000
    evalIO $ setStdGen (mkStdGen rand)
    inps <- evalIO $ replicateM n randomOfShape
    let epochs = 10
        ts     = zip inps (map circle inps)
        trainOptsSgd = TrainingOptions { optimizer = defSGD
                                       , batchSize = 10
                                       , validationFreq = 1
                                       , verbose = Silent
                                       , metrics = []
                                       }
        trainOptsAdam = TrainingOptions { optimizer = defAdam
                                        , batchSize = 10
                                        , validationFreq = 1
                                        , verbose = Silent
                                        , metrics = []
                                        }
    losses  <- evalIO $ mapM (getLosses trainOptsSgd ts rand)  [1..epochs]
    losses' <- evalIO $ mapM (getLosses trainOptsAdam ts rand) [1..epochs]
    -- Perform linear regression on the log of the losses
    let m         = linearRegression epochs losses
        m'        = linearRegression epochs losses'
    -- We expect the loss to decrease over time
    assert $ m < 0
    assert $ m' < 0
  where
    getLosses opts trainData s i = do
      setStdGen (mkStdGen s)
      net <- fit trainData [] opts i quadratic' :: IO FFNet
      let total = foldr (\(x, y) a -> let out = runNet net x in a + (quadratic out y)) 0.0 trainData
          loss  = total / (fromIntegral $ length trainData)
      return loss


prop_training_improves_accuracy :: Property
prop_training_improves_accuracy = withTests 10 $ property $ do
  let n = 1000
  rand <- forAll $ Gen.int $ Range.linear 1 10000
  evalIO $ setStdGen (mkStdGen rand)
  inps <- evalIO $ replicateM n randomOfShape
  let epochs = 20
      ts     = zip inps (map circle inps)
      trainOptsSgd = TrainingOptions { optimizer = defSGD
                                     , batchSize = 10
                                     , validationFreq = 1
                                     , verbose = Silent
                                     , metrics = []
                                     }
      trainOptsAdam = TrainingOptions { optimizer = defAdam
                                      , batchSize = 10
                                      , validationFreq = 1
                                      , verbose = Silent
                                      , metrics = []
                                      }
  accuracies  <- evalIO (mapM (getAccuracy trainOptsSgd ts rand) [1..epochs]  :: IO [Int])
  accuracies' <- evalIO (mapM (getAccuracy trainOptsAdam ts rand) [1..epochs] :: IO [Int])
  -- Perform linear regression on the log of the accuracies
  let acc = map fromIntegral accuracies   :: [Double]
      acc' = map fromIntegral accuracies' :: [Double]
      m  = linearRegression epochs acc
      m' = linearRegression epochs acc'
  -- We expect the accuracy to increase over time
  assert $ m > 0
  assert $ m' > 0
  where
    getAccuracy opts trainData s i = do
      setStdGen (mkStdGen s)
      net <- fit trainData [] opts i quadratic' :: IO FFNet
      inps <- replicateM 100 randomOfShape
      let unshape :: S ('D1 1) -> RealNum
          unshape  = \(S1D x) -> (H.unwrap x) `D.atIndex` 0
          expected = map (unshape . circle) inps
          actual   = map (unshape . runNet net) inps
      let accuracy = length $ filter (<= 0.2) $ zipWith (\x y -> abs (x - y)) expected actual
      return accuracy

tests :: IO Bool
tests = checkParallel $$(discover)
