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
{-|
Module      : Grenade.Layers.LRN
Description : LRN layer
Maintainer  : Aprova Varshney
License     : BSD2
Stability   : experimental

LRN is a non trainable layer which normalises the output of a neuron based on the values in a 
neighborhood around it, especially useful when non-normalised activation functions such as ReLU are utilised.
-}

module Grenade.Layers.LRN 
  ( 
  -- * Layer Definition  
    LRN(..)
  ) where

import           Control.DeepSeq
import           GHC.TypeLits
import           Data.Proxy
import           Data.Serialize

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
  LRN :: ( KnownSymbol a, ValidRealNum a
         , KnownSymbol b, ValidRealNum b
         , KnownSymbol k, ValidRealNum k
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

instance ( KnownSymbol a, ValidRealNum a
         , KnownSymbol b, ValidRealNum b
         , KnownSymbol k, ValidRealNum k
         , KnownNat n
         ) => RandomLayer (LRN a b k n) where
  createRandomWith _ _ = return LRN 

instance ( KnownSymbol a, ValidRealNum a
         , KnownSymbol b, ValidRealNum b
         , KnownSymbol k, ValidRealNum k
         , KnownNat n
         ) => Serialize (LRN a b k n) where
  put _ = return ()
  get   = return LRN

instance ( KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (LRN a b k n) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols channels) where
  
  type Tape (LRN a b k n) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols channels) 
    = (L (inputRows * channels) inputCols, L (inputRows * channels) inputCols)

  runForwards LRN (S3D input) = ((input, sums), S3D normalized)
    where
      normalized :: L (inputRows * channels) inputCols
      normalized = build buildNormalizedMatrix
      
      ex = extract input :: Matrix RealNum
      channels = fromIntegral $ natVal (Proxy :: Proxy channels)  :: Int
      rows     = fromIntegral $ natVal (Proxy :: Proxy inputRows) :: Int

      n  = natVal (Proxy :: Proxy n)
      a  = (read $ symbolVal (Proxy :: Proxy a)) :: RealNum
      b  = (read $ symbolVal (Proxy :: Proxy b)) :: RealNum
      k  = (read $ symbolVal (Proxy :: Proxy k)) :: RealNum

      sums  = build buildSums
      sums' = extract sums
      
      buildSums :: RealNum -> RealNum -> RealNum
      buildSums i j = k + a * summation
        where
          i' = floor i :: Int
          j' = floor j :: Int
          ch = i' `div` rows :: Int
          ro = i' `mod` rows :: Int
          co = j'
          
          sub = floor ((fromIntegral n) / 2     :: RealNum)
          add = floor ((fromIntegral n - 1) / 2 :: RealNum)
          lower = maximum [0, ch - sub]
          upper = minimum [channels - 1, ch + add]
          summation = sum [ (ex `LA.atIndex` (q * rows + ro, co)) ** 2 | q <- [lower..upper]]

      -- Calculates the normalised values for each cell of the input based
      -- on the cross-channel sums of the surrounding cells
      buildNormalizedMatrix :: RealNum -> RealNum -> RealNum
      buildNormalizedMatrix i j = val
        where
          i' = floor i :: Int
          j' = floor j :: Int
          
          f = ex `LA.atIndex` (i', j')
          val = f / den
          
          den  = den' ** b
          den' = sums' `LA.atIndex` (i', j')

  runBackwards LRN (input, sums) (S3D err) = ((), S3D err')
    where
      ex    = extract input :: Matrix RealNum
      exr   = extract err   :: Matrix RealNum
      sums' = extract sums  :: Matrix RealNum
      
      channels = fromIntegral $ natVal (Proxy :: Proxy channels)  :: Int
      rows     = fromIntegral $ natVal (Proxy :: Proxy inputRows) :: Int
      
      n  = natVal (Proxy :: Proxy n)
      a  = (read $ symbolVal (Proxy :: Proxy a)) :: RealNum
      b  = (read $ symbolVal (Proxy :: Proxy b)) :: RealNum
      
      err' = build buildErrorMatrix
      
      buildErrorMatrix :: RealNum -> RealNum -> RealNum
      buildErrorMatrix i j = res
        where
          i' = floor i :: Int
          j' = floor j :: Int
          ch = i' `div` rows
          ro = i' `mod` rows
          co = j'

          sub = floor ((fromIntegral n) / 2     :: RealNum)
          add = floor ((fromIntegral n - 1) / 2 :: RealNum)
          lower = maximum [0, ch - sub]
          upper = minimum [channels - 1, ch + add]

          s  = (ex `LA.atIndex` (i', j')) * s'
          s' = sum [ (ex `LA.atIndex` (q * rows + ro, co)) * (exr `LA.atIndex` (q * rows + ro, co)) | q <- [lower..upper] ]
          c  = sums' `LA.atIndex` (i', j')
          
          res = (exr `LA.atIndex` (i', j')) * (c ** (-b)) - 2 * b * a * (c ** (-b - 1)) * s

instance OnnxOperator (LRN a b k n) where
  onnxOpTypeNames _ = ["LRN"]

instance ( KnownSymbol a, ValidRealNum a
         , KnownSymbol b, ValidRealNum b
         , KnownSymbol k, ValidRealNum k
         , KnownNat n
         )  => OnnxLoadable (LRN a b k n) where
  loadOnnxNode _ node = do
    hasMatchingRealNum node (Proxy :: Proxy a) "alpha"
    hasMatchingRealNum node (Proxy :: Proxy b) "beta"
    hasMatchingRealNum node (Proxy :: Proxy k) "bias"
    hasMatchingInt     node (Proxy :: Proxy n) "size"
    return LRN
    
