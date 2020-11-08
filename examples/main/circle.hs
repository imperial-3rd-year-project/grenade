{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
import           Control.Monad
import           Control.Monad.Random
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static as SA

import           Grenade

{--
  Learning circles
  ================
    This simple example shows how to create a
    network which learns the shape of a circle.
--}

{--
  The Network type
  ================
    For this simple example, we will use a network consisting of
    fully connected layers. There are two hidden layers with 40 neurons
    and 10 neurons respectively. We use the tanh, relu and logit 
    activation functions.
    Given an (x, y) coordinate, the output is the probability the 
    coordinate is inside the circle.
--}
type FFNet 
  = Network 
    '[ FullyConnected 2 40, Tanh, 
       FullyConnected 40 10, Relu, 
       FullyConnected 10 1, Logit 
     ]
    '[ 'D1 2,          -- Input
       'D1 40, 'D1 40, -- Tanh fully connected
       'D1 10, 'D1 10, -- Relu fully connected
       'D1 1, 'D1 1    -- Logit fully connected
     ]

{--
  A helper function for displaying the circles.
--}
drawCircle :: FFNet -> IO ()
drawCircle network = do
    let testIns = [ [ (x,y)  | x <- [0..50] ] | y <- [0..20] ]
        outMat  = (map . map) renderCoord testIns
    putStrLn $ unlines outMat

  where
    render n'  | n' <= 0.2  = ' '
               | n' <= 0.4  = '.'
               | n' <= 0.6  = '-'
               | n' <= 0.8  = '='
               | otherwise = '#'

    normx :: S ('D1 1) -> RealNum
    normx (S1D r) = SA.mean r

    renderCoord (x, y) = (render . normx) $ runNet network (S1D $ SA.vector [x / 25 - 1, y / 10 - 1])

{--
  This is the function we are attempting to learn. 
  Given a vector (x, y), we check if the distance from the origin
  is less than a radius of 0.5, and return 1 if the coordinate is
  inside the circle and 0 otherwise.
--}
circle :: S ('D1 2) -> S ('D1 1)
circle (S1D v) =
  if v `inCircle` (fromRational 0, 0.5)
     then S1D $ fromRational 1
     else S1D $ fromRational 0
  where
    inCircle :: KnownNat n => SA.R n -> (SA.R n, RealNum) -> Bool
    u `inCircle` (o, r) = SA.norm_2 (u - o) <= r
    

{--
  Training and Execution
  ================
    Now we train our network and run it.
--}
main :: IO ()
main = do
    -- Train on 100000 examples, randomly generated with a uniform 
    -- distribution.
    let n = 100000
    inps <- replicateM n $ do
      s  <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
    -- For our network, we will use stochastic gradient descent with a batch
    -- size of 1. 
    let options = TrainingOptions { optimizer      = OptSGD 0.01 0.9 0.0001
                                  , batchSize      = 1
                                  , validationFreq = 1
                                  , verbose        = Full 
                                  , metrics        = [Quadratic]
                                  }
    -- Now we train the network with the fit function.
    -- We pass in a list of tuples of inputs and expected outputs,
    -- and use the quadratic loss function. We train for one epoch.
    -- There is no validation data, hence we pass in an empty list.
    net <- fit (zip inps (map circle inps)) [] options 1 quadratic'

    -- Now we visualise what our network has learned.
    drawCircle net
