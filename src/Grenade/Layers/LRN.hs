{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE OverloadedLabels      #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module Grenade.Layers.LRN 
  ( LRN(..)
  ) where

import           Control.DeepSeq
import           GHC.TypeLits
import           Data.Proxy
import           Data.Serialize

import qualified Numeric.LinearAlgebra.Data as NLD
import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static

import           Grenade.Core
import           Grenade.Types
import           Grenade.Onnx
import           Data.Kind (Type)
import           Numeric.LinearAlgebra (Matrix)
import           Grenade.Utils.Symbols
import           Data.ProtoLens.Labels ()

data LRN :: Symbol -> Symbol -> Symbol -> Nat -> Type where
  LRN :: ( KnownSymbol a, ValidDouble a
         , KnownSymbol b, ValidDouble b
         , KnownSymbol k, ValidDouble k
         , KnownNat n
         ) => LRN a b k n

instance NFData (LRN a b k n) where
  rnf LRN = ()

instance Show (LRN a b k n) where
  show LRN 
    = "LRN " 
    ++ (symbolVal (Proxy :: Proxy a)) ++ " "
    ++ (symbolVal (Proxy :: Proxy b)) ++ " "
    ++ (symbolVal (Proxy :: Proxy k)) ++ " "
    ++ show (natVal (Proxy :: Proxy n))

instance UpdateLayer (LRN a b k n) where
  type Gradient (LRN a b k n) = ()
  runUpdate _ x _ = x
  reduceGradient _ = ()

instance ( KnownSymbol a, ValidDouble a
         , KnownSymbol b, ValidDouble b
         , KnownSymbol k, ValidDouble k
         , KnownNat n
         ) => RandomLayer (LRN a b k n) where
  createRandomWith _ _ = return LRN 

instance ( KnownSymbol a, ValidDouble a
         , KnownSymbol b, ValidDouble b
         , KnownSymbol k, ValidDouble k
         , KnownNat n
         ) => Serialize (LRN a b k n) where
  put _ = return ()
  get   = return LRN

instance ( KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (LRN a b k n) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols channels) where
  
  type Tape (LRN a b k n) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols channels) 
    = (S ('D3 inputRows inputCols channels), Matrix RealNum)

  runForwards LRN (S3D input) = ((S3D input, sums), S3D normalized)
    where
      normalized :: L (inputRows * channels) inputCols
      normalized = build buildMat

      ex = extract input :: Matrix RealNum

      sums :: Matrix RealNum 
      sums = NLD.build (n, m) buildSums
        where
          h = natVal (Proxy :: Proxy inputRows)
          w = natVal (Proxy :: Proxy inputCols)
          c = natVal (Proxy :: Proxy channels)
          n = fromInteger $ h * c
          m = fromInteger w

      buildSums :: Double -> Double -> Double
      buildSums row col = c
        where
          i = floor $ row / (fromIntegral rs)
          x = floor $ row - (fromIntegral (i * rs))
          y = floor col
          c = k + a * summation
          sub = floor ((fromIntegral n) / 2     :: Double)
          add = floor ((fromIntegral n - 1) / 2 :: Double)
          lower = maximum [0, i - sub]
          upper = minimum [cs - 1, i + add]
          summation = sum [ (elem j x y) ** 2 | j <- [lower..upper]]

      buildMat :: Double -> Double -> Double
      buildMat row col = num / den
        where
          i = floor $ row / (fromIntegral rs)
          x = floor $ row - (fromIntegral (i * rs))
          y = floor col
          num = elem i x y
          den = c ** b
          c = access sums cs i x y

      elem :: Int -> Int -> Int -> Double
      elem = access ex cs

      cs = fromIntegral $ natVal (Proxy :: Proxy channels)
      rs = fromIntegral $ natVal (Proxy :: Proxy inputRows)
      n  = natVal (Proxy :: Proxy n)
      a  = (read $ symbolVal (Proxy :: Proxy a)) :: Double
      b  = (read $ symbolVal (Proxy :: Proxy b)) :: Double
      k  = (read $ symbolVal (Proxy :: Proxy k)) :: Double

  runBackwards LRN (S3D input, sums) (S3D err) = ((), S3D err')
    where
      ex   = extract input  :: Matrix RealNum
      exr  = extract err    :: Matrix RealNum
      err' = build buildMat

      buildMat :: Double -> Double -> Double
      buildMat row col = exrixy * (c ** (-b)) - 2 * b * a * (c ** (-b - 1)) * sum'
        where
          i = floor $ row / (fromIntegral rs)
          x = floor $ row - (fromIntegral (i * rs))
          y = floor col
          sub = floor ((fromIntegral n) / 2     :: Double)
          add = floor ((fromIntegral n - 1) / 2 :: Double)
          lower = maximum [0, i - sub]
          upper = minimum [cs - 1, i + add]
          c = access sums cs i x y
          sum' = elem i x y * sum [ elem j x y * elem' j | j <- [lower..upper]]

          exrixy = access exr cs i x y
          elem' j = access exr cs j x y

      elem :: Int -> Int -> Int -> Double
      elem = access ex cs

      cs = fromIntegral $ natVal (Proxy :: Proxy channels)
      rs = fromIntegral $ natVal (Proxy :: Proxy inputRows)
      n  = natVal (Proxy :: Proxy n)
      a  = (read $ symbolVal (Proxy :: Proxy a)) :: Double
      b  = (read $ symbolVal (Proxy :: Proxy b)) :: Double

access :: Matrix RealNum -> Int -> Int -> Int -> Int -> Double
access ex cs j x y = ex LA.! (block + x) LA.! y
  where
    block = j * cs

instance OnnxOperator (LRN a b k n) where
  onnxOpTypeNames _ = ["LRN"]

instance ( KnownSymbol a, ValidDouble a
         , KnownSymbol b, ValidDouble b
         , KnownSymbol k, ValidDouble k
         , KnownNat n
         )  => OnnxLoadable (LRN a b k n) where
  loadOnnxNode _ node = do
    hasMatchingDouble node (Proxy :: Proxy a) "alpha"
    hasMatchingDouble node (Proxy :: Proxy b) "beta"
    hasMatchingDouble node (Proxy :: Proxy k) "bias"
    hasMatchingInt    node (Proxy :: Proxy n) "size"
    return LRN
    
