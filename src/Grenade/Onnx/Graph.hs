{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}

module Grenade.Onnx.Graph 
  ( readInitializer
  )
  where

import Data.ProtoLens.Labels ()
import qualified Proto.Onnx as P
import Lens.Micro
import qualified Data.Text as T
import Data.Maybe (listToMaybe)

readInitializer :: P.GraphProto -> T.Text -> Maybe ([Int], [Float])
readInitializer graph name = listToMaybe (filter (\tensor -> tensor ^. #name == name) inits) >>= retrieve
  where 
    inits = graph ^. #initializer
    retrieve tensor = case toEnum (fromIntegral (tensor ^. #dataType)) of
                        P.TensorProto'FLOAT -> Just (map fromIntegral (tensor ^. #dims), tensor ^. #floatData)
                        _                   -> Nothing
