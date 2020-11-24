module Grenade.Onnx.OnnxOperator (OnnxOperator (..)) where

import           Data.Proxy
import qualified Data.Text as T

class OnnxOperator a where
  onnxOpTypeNames :: Proxy a -> [T.Text]

