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

-- A simple function for displaying the circle
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

-- This is the function we will attempt to learn
circle :: S ('D1 2) -> S ('D1 1)
circle (S1D v) =
  if v `inCircle` (fromRational 0, 0.5)
     then S1D $ fromRational 1
     else S1D $ fromRational 0
  where
    inCircle :: KnownNat n => SA.R n -> (SA.R n, RealNum) -> Bool
    u `inCircle` (o, r) = SA.norm_2 (u - o) <= r
    

main :: IO ()
main = do
    let n = 100000
    inps <- replicateM n $ do
      s  <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
    let options = TrainingOptions { optimizer      = OptSGD 0.01 0.9 0.0001
                                  , batchSize      = 1
                                  , validationFreq = 1
                                  , verbose        = Full 
                                  , metrics        = []
                                  }
    -- No validation data
    net <- fit (zip inps (map circle inps)) [] options 1 quadratic'
    drawCircle net
