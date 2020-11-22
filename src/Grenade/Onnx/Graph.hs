{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE DeriveFunctor       #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedLabels    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Grenade.Onnx.Graph
  ( readInitializerMatrix
  , readInitializerVector
  , readInitializerTensorIntoMatrix
  , readIntAttribute
  , filterIntAttribute
  , readIntsAttribute
  , filterIntsAttribute
  , readDoubleAttribute
  , filterDoubleAttribute
  , doesNotHaveAttribute
  )
  where

import           Control.Monad                (guard)

import           Data.List                    (find)
import qualified Data.Map.Strict              as Map
import           Data.Maybe                   (isNothing)
import           Data.ProtoLens.Labels        ()
import           Data.Proxy
import qualified Data.Text                    as T

import           GHC.Float                    (float2Double)
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static

import           Lens.Micro
import qualified Proto.Onnx                   as P

readAttribute :: T.Text -> P.NodeProto -> Maybe P.AttributeProto
readAttribute attribute node = find ((== attribute) . (^. #name)) (node ^. #attribute)

readIntAttribute :: T.Text -> P.NodeProto -> Maybe Int
readIntAttribute attribute node = readAttribute attribute node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'INT -> Just $ fromIntegral $ attribute ^. #i
                           _                      -> Nothing

filterIntAttribute :: T.Text -> (Int -> Bool) -> P.NodeProto -> Maybe ()
filterIntAttribute attribute pred node = readIntAttribute attribute node >>= guard . pred

readDoubleAttribute :: T.Text -> P.NodeProto -> Maybe Double
readDoubleAttribute attribute node = readAttribute attribute node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'FLOAT -> Just $ float2Double $ attribute ^. #f
                           _                      -> Nothing

filterDoubleAttribute :: T.Text -> (Double -> Bool) -> P.NodeProto -> Maybe ()
filterDoubleAttribute attribute pred node = readDoubleAttribute attribute node >>= guard . pred

readIntsAttribute :: T.Text -> P.NodeProto -> Maybe [Int]
readIntsAttribute attribute node = readAttribute attribute node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'INTS -> Just $ map fromIntegral $ attribute ^. #ints
                           _                     -> Nothing

filterIntsAttribute :: T.Text -> ([Int] -> Bool) -> P.NodeProto -> Maybe ()
filterIntsAttribute attribute pred node = readIntsAttribute attribute node >>= guard . pred

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
