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

module Grenade.Onnx.Model (loadOnnxModel, loadOnnxModel') where

import qualified Data.ByteString as B
import           Data.Foldable           (toList)

import           Data.ProtoLens.Labels   ()
import           Data.ProtoLens.Encoding (decodeMessage)

import           Grenade.Onnx.Graph
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.OnnxLoadFailure
import           Grenade.Onnx.Utils

-- | Load model from an ONNX model written to file
loadOnnxModel :: OnnxLoadable a => FilePath -> IO (Either OnnxLoadFailure a)
loadOnnxModel path = do
  modelContent <- B.readFile path
  return $! loadOnnxModel' modelContent
{-# INLINABLE loadOnnxModel #-}

-- | Load network from an ONNX model serialized to ByteString
loadOnnxModel' :: OnnxLoadable a
               => B.ByteString
               -> Either OnnxLoadFailure a
loadOnnxModel' msg =
  case decodeMessage msg of
    Left err -> loadFailureReason $ show err
    Right model -> do
      (graphProto, graph) <- generateGraph model
      let initMap = generateInitializerMap graphProto
      (network, lastSucc, graph') <- loadOnnx initMap graph
      case toList graph' of
        []      -> return network
        (x : _) -> loadFailure "Node not loaded" lastSucc (Just x) []
{-# INLINABLE loadOnnxModel' #-}
