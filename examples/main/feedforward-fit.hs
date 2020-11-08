{-# LANGUAGE BangPatterns        #-}
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


type FFNet = Network '[ FullyConnected 2 40, Tanh, FullyConnected 40 10, Relu, FullyConnected 10 1, Logit ]
                     '[ 'D1 2, 'D1 40, 'D1 40, 'D1 10, 'D1 10, 'D1 1, 'D1 1]

inCircle :: KnownNat n => SA.R n -> (SA.R n, RealNum) -> Bool
v `inCircle` (o, r) = SA.norm_2 (v - o) <= r
    
netScore :: FFNet -> IO ()
netScore network = do
    let testIns = [ [ (x,y)  | x <- [0..50] ]
                             | y <- [0..20] ]
        outMat  = fmap (fmap (\(x,y) -> (render . normx) $ runNet network (S1D $ SA.vector [x / 25 - 1,y / 10 - 1]))) testIns
    putStrLn $ unlines outMat

  where
    render n'  | n' <= 0.2  = ' '
               | n' <= 0.4  = '.'
               | n' <= 0.6  = '-'
               | n' <= 0.8  = '='
               | otherwise = '#'

    normx :: S ('D1 1) -> RealNum
    normx (S1D r) = SA.mean r

-- This is the function we will attempt to learn
f :: S ('D1 2) -> S ('D1 1)
f (S1D v) =
  if v `inCircle` (fromRational 0.33, 0.33)  || v `inCircle` (fromRational (-0.33), 0.33)
     then S1D $ fromRational 1
     else S1D $ fromRational 0

main :: IO ()
main = do
    let examples = 200000
    inps :: [S ('D1 2)] <- replicateM examples $ do
      s  <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
    let options = TrainingOptions { optimizer      = OptSGD 0.01 0.9 0.0001
                                    , batchSize      = 1
                                    , validationFreq = 1
                                    , verbose        = Full 
                                    , metrics        = []
                                  }
    -- No validation data
    net <- fit (zip inps (map f inps)) [] options 1 quadratic'
    netScore net
