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
  , readIntsAttribute
  )
  where

import Data.ProtoLens.Labels ()
import qualified Proto.Onnx as P
import Lens.Micro
import qualified Data.Text as T
import Data.Maybe (listToMaybe)

import qualified Data.Map.Strict                as Map
import           Numeric.LinearAlgebra.Static
import           GHC.TypeLits
import           GHC.Float (float2Double)
import           Data.Proxy

readAttribute :: T.Text -> P.NodeProto -> Maybe P.AttributeProto
readAttribute attribute node = listToMaybe $ filter ((== attribute) . (^. #name)) $ node ^. #attribute

readIntsAttribute :: T.Text -> P.NodeProto -> Maybe [Int]
readIntsAttribute attribute node = map fromIntegral <$> (^. #ints) <$> readAttribute attribute node

readInitializer :: Map.Map T.Text P.TensorProto -> T.Text -> Maybe ([Int], [Double])
readInitializer inits name = Map.lookup name inits >>= retrieve
  where 
    retrieve tensor = case toEnum (fromIntegral (tensor ^. #dataType)) of
                        P.TensorProto'FLOAT -> Just (map fromIntegral (tensor ^. #dims), map float2Double (tensor ^. #floatData))
                        _                   -> Nothing

readInitializerMatrix :: (KnownNat r, KnownNat c) => Map.Map T.Text P.TensorProto -> T.Text -> Maybe (L r c)
readInitializerMatrix inits name = readInitializer inits name >>= readMatrix

readInitializerVector :: KnownNat r => Map.Map T.Text P.TensorProto -> T.Text -> Maybe (R r)
readInitializerVector inits name = readInitializer inits name >>= readVector

readMatrix :: forall r c . (KnownNat r, KnownNat c) => ([Int], [Double]) -> Maybe (L r c)
readMatrix ([rows, cols], vals)
  | rows == neededRows && cols == neededCols = Just (matrix vals)
  where
    neededRows = fromIntegral $ natVal (Proxy :: Proxy r)
    neededCols = fromIntegral $ natVal (Proxy :: Proxy c)
readMatrix _ = Nothing

readVector :: forall r . KnownNat r => ([Int], [Double]) -> Maybe (R r)
readVector ([rows], vals)
  | rows == neededRows = Just (vector vals)
  where
    neededRows = fromIntegral $ natVal (Proxy :: Proxy r)
readVector _ = Nothing
