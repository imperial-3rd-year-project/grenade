{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE UndecidableInstances #-}

module Grenade.Onnx.OnnxLoadable where


import           Control.Applicative
import           Control.Monad
import qualified Data.Map.Strict     as Map
import qualified Data.Text           as T
import           Lens.Micro

import           Grenade.Onnx.Onnx

import qualified Proto.Onnx          as P

class OnnxLoadable a where
  loadOnnx :: Map.Map T.Text P.TensorProto -> SPG 'S P.NodeProto -> Maybe (a, SPG 'S P.NodeProto)


hasType :: Alternative m => P.NodeProto -> T.Text -> m ()
hasType node typeString = guard $ typeString == (node ^. #opType)
