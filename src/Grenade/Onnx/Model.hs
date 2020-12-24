{-# LANGUAGE DeriveFunctor       #-}
{-# LANGUAGE StandaloneDeriving  #-}
{-# LANGUAGE OverloadedLabels    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE FlexibleContexts    #-}

module Grenade.Onnx.Model (loadOnnxModel) where

import qualified Data.ByteString as B

import           Data.ProtoLens.Labels   ()
import           Data.ProtoLens.Encoding (decodeMessage)
import           Lens.Micro              ((^.), _1)
import qualified Proto.Onnx as P

import           Grenade.Onnx.Graph
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.OnnxLoadFailure
import           Grenade.Onnx.Utils

readOnnxModel :: FilePath -> IO (Either OnnxLoadFailure P.ModelProto)
readOnnxModel path = do
  modelContent <- B.readFile path
  let (model :: Either String P.ModelProto) = decodeMessage modelContent
  case model of
    Left err -> return . loadFailureReason $ show err
    Right modelProto -> return (Right modelProto)
{-# INLINE readOnnxModel #-}

loadOnnxModel :: OnnxLoadable a => FilePath -> IO (Either OnnxLoadFailure a)
loadOnnxModel path = do
  eitherModel <- readOnnxModel path
  return $! do
    model <- eitherModel
    (graphProto, graph) <- generateGraph model
    let initMap = generateInitializerMap graphProto
    (^. _1) <$> loadOnnx initMap graph
{-# INLINABLE loadOnnxModel #-}
