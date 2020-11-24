{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE FlexibleContexts     #-}

module Grenade.Onnx.OnnxLoadFailure
  ( OnnxLoadFailure (..)
  , reason
  , lastSuccessfulNode
  , currentNode
  , expectingNodeTypes
  ) where

import           Control.Applicative ((<|>))
import           Data.Maybe          (fromMaybe)
import           Lens.Micro          ((^.))
import           Lens.Micro.TH

import           Data.ProtoLens.Labels        ()
import qualified Proto.Onnx          as P

data OnnxLoadFailure = OnnxLoadFailure { _reason             :: String
                                       , _lastSuccessfulNode :: Maybe P.NodeProto
                                       , _currentNode        :: Maybe P.NodeProto
                                       , _expectingNodeTypes :: [String]
                                       }
makeLenses ''OnnxLoadFailure

instance Show OnnxLoadFailure where
  show (OnnxLoadFailure reason lastNode failedNode expectedTypes) =
    "Onnx loading failed " ++ fromMaybe "" (showNode failedNodeString failedNode <|> showNode lastNodeString lastNode)
    ++ "; " ++ reason ++ expectedString
    where 
      showNode str = ((++ "'") . ((str ++ "' ") ++) . show . (^. #name) <$>)
      lastNodeString = "after last successfully loaded node"
      failedNodeString = "in node"
      expectedString = case expectedTypes of
                         [] -> ""
                         ts -> "; expecting one of: " ++ show ts
