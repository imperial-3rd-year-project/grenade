{-# LANGUAGE OverloadedStrings #-}

module Grenade.Onnx.Utils where

import           Grenade.Onnx.Graph

import           Control.Monad                (guard)
import           Data.ProtoLens.Labels        ()
import           Data.Proxy
import qualified Data.Text                    as T
import           GHC.TypeLits
import qualified Proto.Onnx                   as P

hasSupportedGroup :: P.NodeProto -> Maybe ()
hasSupportedGroup = filterIntAttribute "group" (== 1)

hasSupportedDilations :: P.NodeProto -> Maybe ()
hasSupportedDilations = filterIntsAttribute "dilations" (all (==1))

hasMatchingShape :: P.NodeProto -> T.Text -> [Integer] -> Maybe ()
hasMatchingShape node attribute dims = filterIntsAttribute attribute (== map fromIntegral dims) node

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
