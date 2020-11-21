{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}

module Grenade.Onnx.Graph 
  ( readInitializerMatrix
  , readInitializerVector
  , readInitializerTensorIntoMatrix
  , readIntAttribute
  , readIntsAttribute
  , readDoubleAttribute
  , doesNotHaveAttribute
  , hasCorrectPadding
  )
  where

import Data.ProtoLens.Labels ()
import qualified Proto.Onnx as P
import Lens.Micro
import qualified Data.Text as T
import Data.Maybe (isNothing, listToMaybe)

import           Control.Monad (guard)
import qualified Data.Map.Strict                as Map
import           Data.Proxy
import           GHC.TypeLits
import           GHC.Float (float2Double)
import           Numeric.LinearAlgebra.Static

readAttribute :: T.Text -> P.NodeProto -> Maybe P.AttributeProto
readAttribute attribute node = listToMaybe $ filter ((== attribute) . (^. #name)) $ node ^. #attribute

readIntAttribute :: T.Text -> P.NodeProto -> Maybe Int
readIntAttribute attribute node = readAttribute attribute node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'INT -> Just $ fromIntegral $ attribute ^. #i
                           _                      -> Nothing

readDoubleAttribute :: T.Text -> P.NodeProto -> Maybe Double
readDoubleAttribute attribute node = readAttribute attribute node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'FLOAT -> Just $ float2Double $ attribute ^. #f
                           _                      -> Nothing

readIntsAttribute :: T.Text -> P.NodeProto -> Maybe [Int]
readIntsAttribute attribute node = readAttribute attribute node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'INTS -> Just $ map fromIntegral $ attribute ^. #ints
                           _                     -> Nothing

doesNotHaveAttribute :: P.NodeProto -> T.Text -> Maybe ()
doesNotHaveAttribute node attribute = guard $ isNothing $ readAttribute attribute node

readInitializer :: Map.Map T.Text P.TensorProto -> T.Text -> Maybe ([Int], [Double])
readInitializer inits name = Map.lookup name inits >>= retrieve
  where 
    retrieve tensor = case toEnum (fromIntegral (tensor ^. #dataType)) of
                        P.TensorProto'FLOAT -> Just (map fromIntegral (tensor ^. #dims), map float2Double (tensor ^. #floatData))
                        _                   -> Nothing

readInitializerMatrix :: (KnownNat r, KnownNat c) => Map.Map T.Text P.TensorProto -> T.Text -> Maybe (L r c)
readInitializerMatrix inits name = readInitializer inits name >>= readMatrix

readInitializerTensorIntoMatrix :: (KnownNat r, KnownNat c) => Map.Map T.Text P.TensorProto -> T.Text -> Maybe (L r c)
readInitializerTensorIntoMatrix inits name = readInitializer inits name >>= readTensorIntoMatrix

readInitializerVector :: KnownNat r => Map.Map T.Text P.TensorProto -> T.Text -> Maybe (R r)
readInitializerVector inits name = readInitializer inits name >>= readVector

readMatrix :: forall r c . (KnownNat r, KnownNat c) => ([Int], [Double]) -> Maybe (L r c)
readMatrix ([rows, cols], vals)
  | neededRows == rows && neededCols == cols = Just (matrix vals)
  where
    neededRows = fromIntegral $ natVal (Proxy :: Proxy r)
    neededCols = fromIntegral $ natVal (Proxy :: Proxy c)
readMatrix _ = Nothing

readTensorIntoMatrix :: forall r c . (KnownNat r, KnownNat c) => ([Int], [Double]) -> Maybe (L r c)
readTensorIntoMatrix (rows : cols, vals)
  | neededRows == rows && neededCols == product cols = Just (matrix vals)
  where
    neededRows = fromIntegral $ natVal (Proxy :: Proxy r)
    neededCols = fromIntegral $ natVal (Proxy :: Proxy c)
readTensorIntoMatrix _ = Nothing

readVector :: forall r . KnownNat r => ([Int], [Double]) -> Maybe (R r)
readVector ([rows], vals)
  | rows == neededRows = Just (vector vals)
  where
    neededRows = fromIntegral $ natVal (Proxy :: Proxy r)
readVector _ = Nothing

hasCorrectPadding :: (KnownNat padLeft, KnownNat padRight, KnownNat padTop, KnownNat padBottom)
                  => P.NodeProto -> Proxy padLeft -> Proxy padRight -> Proxy padTop -> Proxy padBottom -> Maybe ()
hasCorrectPadding node ppl ppr ppt ppb 
  = let left   = fromIntegral $ natVal ppl
        right  = fromIntegral $ natVal ppr
        top    = fromIntegral $ natVal ppt
        bottom = fromIntegral $ natVal ppb
     in case readIntsAttribute "pads" node of
          Just [left', top', right', bottom'] -> guard (left == left' && top == top' && right == right' && bottom == bottom')
          Nothing                             -> guard (left == 0     && top == 0    && right == 0      && bottom == 0)
          _                                   -> Nothing
