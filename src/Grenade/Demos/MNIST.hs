
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# LANGUAGE TypeFamilies        #-}

{-# LANGUAGE OverloadedStrings   #-}

module Grenade.Demos.MNIST where

import qualified Data.ByteString              as B
import           Data.List                    (maximumBy)
import           Data.Maybe                   (fromMaybe)
import           Data.Ord                     (comparing)
import           Data.Serialize               (Get, runGet, get)
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Unboxed          as UB
import qualified Data.Vector                  as DV
import           Data.Word


import           Grenade
import           Grenade.Layers.Internal.Shrink (shrink_2d_rgba)

import qualified Numeric.LinearAlgebra.Static as SA


netLoad :: FilePath -> IO MNIST
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet (get :: Get MNIST) modelData

runNet'' :: MNIST -> UB.Vector Word8 -> Int -> String
runNet'' net vec = runNet' net (DV.convert vec) 

runNet' :: MNIST -> V.Vector Word8 -> Int -> String
runNet' net m d = (\(S1D ps) -> let (p, i) = (getProb . V.toList) (SA.extract ps)
                              in "This number is " ++ show i ++ " with probability " ++ (show (round (p * 100) :: Int)) ++ "%") $ runNet net (conv m)
  where
    getProb :: (Ord a) => [a] -> (a, Int)
    getProb xs = maximumBy (comparing fst) (Prelude.zip xs [0..])

    conv :: V.Vector Word8 -> S ('D2 28 28)
    conv m = S2D $ fromMaybe (error "") $ SA.create $ shrink_2d_rgba d d 28 28 m 

type MNIST
  = Network
    '[ Convolution 'WithoutBias 'NoPadding 1 10 5 5 1 1
     , Pooling 2 2 2 2
     , Relu
     , Convolution 'WithoutBias 'NoPadding 10 16 5 5 1 1
     , Pooling 2 2 2 2
     , Reshape
     , Relu
     , FullyConnected 256 80
     , Logit
     , FullyConnected 80 10
     , Logit]
    '[ 'D2 28 28
     , 'D3 24 24 10
     , 'D3 12 12 10
     , 'D3 12 12 10
     , 'D3 8 8 16
     , 'D3 4 4 16
     , 'D1 256
     , 'D1 256
     , 'D1 80
     , 'D1 80
     , 'D1 10
     , 'D1 10]
