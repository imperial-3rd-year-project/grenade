{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE OverloadedLabels    #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}

module Grenade.Onnx.Utils 
  ( hasType
  , loadFailure
  , loadFailureReason
  , loadFailureAttr
  , loadFailureExpecting
  , guardOnnx
  , guardAttr
  , readInitializerMatrix
  , readInitializerVector
  , readInitializerTensorIntoMatrix
  , readIntAttribute
  , filterIntAttribute
  , readIntsAttribute
  , filterIntsAttribute
  , readDoubleAttribute
  , filterDoubleAttribute
  , doesNotHaveAttribute
  , hasSupportedGroup
  , hasSupportedDilations
  , hasMatchingShape
  , hasMatchingInt
  , hasMatchingDouble
  , hasCorrectPadding
  , onnxIncorrectNumberOfInputs
  ) where


import           Control.Monad                (void)
import           Data.Either.Combinators
import qualified Data.Map.Strict              as Map
import           Data.List                    (find)
import           Data.Proxy

import           Grenade.Onnx.OnnxOperator
import           Grenade.Onnx.OnnxLoadFailure

import           Numeric.LinearAlgebra.Static hiding ((<>))

import           GHC.Float                    (float2Double)
import           GHC.TypeLits

import           Data.ProtoLens.Labels        ()
import qualified Data.Text                    as T
import           Lens.Micro                   ((^.))
import qualified Proto.Onnx                   as P


hasType :: forall a. OnnxOperator a => P.NodeProto -> Proxy a -> Either OnnxLoadFailure ()
hasType node a
  | node ^. #opType `elem` expectedTypes = return ()
  | otherwise = void loadFail
  where 
    nodeType = node ^. #opType
    expectedTypes = onnxOpTypeNames a
    loadFail :: Either OnnxLoadFailure (a, b, c)
    loadFail = loadFailureExpecting ("Unexpected node type: " ++ show nodeType) (Just node)

loadFailure :: String -> Maybe P.NodeProto -> Maybe P.NodeProto -> [String] -> Either OnnxLoadFailure a
loadFailure reason last curr types = Left (OnnxLoadFailure reason last curr types)

loadFailureReason :: String -> Either OnnxLoadFailure a
loadFailureReason reason = Left (OnnxLoadFailure reason Nothing Nothing [])

loadFailureAttr :: String -> P.NodeProto -> Either OnnxLoadFailure a
loadFailureAttr reason curr = loadFailure reason Nothing (Just curr) []

loadFailureExpecting :: forall a b c. OnnxOperator a => String -> Maybe P.NodeProto -> Either OnnxLoadFailure (a, b, c)
loadFailureExpecting reason curr = loadFailure reason Nothing curr (show <$> onnxOpTypeNames (Proxy :: Proxy a))

guardOnnx :: String -> Bool -> Either OnnxLoadFailure ()
guardOnnx reason cond
  | cond = return ()
  | otherwise = loadFailureReason reason

guardAttr :: T.Text -> Bool -> Either OnnxLoadFailure ()
guardAttr attribute = guardOnnx ("Attribute '" ++ show attribute ++ "' did not pass predicate")

maybeLoadFailureReason :: String -> Maybe a -> Either OnnxLoadFailure a
maybeLoadFailureReason reason = maybe (loadFailureReason reason) Right

readAttribute :: T.Text -> P.NodeProto -> Either OnnxLoadFailure P.AttributeProto
readAttribute attribute node = maybeLoadFailureReason ("Attribute '" ++ show attribute ++ "' was not found")
                                 (find ((== attribute) . (^. #name)) (node ^. #attribute))

readIntAttribute :: T.Text -> P.NodeProto -> Either OnnxLoadFailure Int
readIntAttribute attributeName node = readAttribute attributeName node >>= retrieve
  where
    retrieve attribute =
      case attribute ^. #type' of
        P.AttributeProto'INT -> Right $ fromIntegral $ attribute ^. #i
        _                    -> loadFailureReason $ "Type of attribute '" ++ show attributeName ++ "' is not int"

filterIntAttribute :: T.Text -> (Int -> Bool) -> P.NodeProto -> Either OnnxLoadFailure ()
filterIntAttribute attribute pred node = readIntAttribute attribute node >>= guardAttr attribute . pred

readDoubleAttribute :: T.Text -> P.NodeProto -> Either OnnxLoadFailure Double
readDoubleAttribute attributeName node = readAttribute attributeName node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'FLOAT -> Right . float2Double $ attribute ^. #f
                           _                      -> loadFailureReason $ 
                                                       "Type of attribute '" ++ show attributeName ++ "' is not float"

filterDoubleAttribute :: T.Text -> (Double -> Bool) -> P.NodeProto -> Either OnnxLoadFailure ()
filterDoubleAttribute attribute pred node = readDoubleAttribute attribute node >>= guardAttr attribute . pred

readIntsAttribute :: T.Text -> P.NodeProto -> Either OnnxLoadFailure [Int]
readIntsAttribute attributeName node = readAttribute attributeName node >>= retrieve
  where
    retrieve attribute = case (attribute ^. #type') of
                           P.AttributeProto'INTS -> Right . map fromIntegral $ attribute ^. #ints
                           _                     -> loadFailureReason $ 
                                                      "Type of attribute '" ++ show attributeName ++ "' is not ints"

filterIntsAttribute :: T.Text -> ([Int] -> Bool) -> P.NodeProto -> Either OnnxLoadFailure ()
filterIntsAttribute attribute pred node = readIntsAttribute attribute node >>= guardAttr attribute . pred

doesNotHaveAttribute :: P.NodeProto -> T.Text -> Either OnnxLoadFailure ()
doesNotHaveAttribute node attribute = guardAttr attribute $ isLeft $ readAttribute attribute node

readInitializer :: Map.Map T.Text P.TensorProto -> T.Text -> Either OnnxLoadFailure ([Int], [Double])
readInitializer inits name = maybeLoadFailureReason ("Initializer '" ++ show name ++ "' does not exist")
                               (Map.lookup name inits) >>= retrieve
  where
    retrieve tensor = case toEnum (fromIntegral (tensor ^. #dataType)) of
                        P.TensorProto'FLOAT -> Right (map fromIntegral (tensor ^. #dims), map float2Double (tensor ^. #floatData))
                        _                   -> loadFailureReason $ 
                                                 "Type of attribute '" ++ show name ++ "' is not float"

readInitializerMatrix :: (KnownNat r, KnownNat c) => Map.Map T.Text P.TensorProto -> T.Text -> Either OnnxLoadFailure (L r c)
readInitializerMatrix inits name = do
  initializer <- readInitializer inits name 
  maybeLoadFailureReason ("Failed to read initializer '" ++ show name ++ "' as matrix") (readMatrix initializer)

readInitializerTensorIntoMatrix :: (KnownNat r, KnownNat c) => Map.Map T.Text P.TensorProto -> T.Text -> Either OnnxLoadFailure (L r c)
readInitializerTensorIntoMatrix inits name = do
  initializer <- readInitializer inits name
  maybeLoadFailureReason ("Failed to read initializer tensor '" ++ show name ++ "' as matrix") (readTensorIntoMatrix initializer)

readInitializerVector :: KnownNat r => Map.Map T.Text P.TensorProto -> T.Text -> Either OnnxLoadFailure (R r)
readInitializerVector inits name = do
  initializer <- readInitializer inits name
  maybeLoadFailureReason ("Failed to read initializer '" ++ show name ++ "' as vector") (readVector initializer)

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

hasSupportedGroup :: P.NodeProto -> Either OnnxLoadFailure ()
hasSupportedGroup node = (node `doesNotHaveAttribute` "group")
                      <> (filterIntAttribute "group" (== 1) node)

hasSupportedDilations :: P.NodeProto -> Either OnnxLoadFailure ()
hasSupportedDilations node = (node `doesNotHaveAttribute` "dilations")
                          <> (filterIntsAttribute "dilations" (all (==1)) node)

hasMatchingShape :: P.NodeProto -> T.Text -> [Integer] -> Either OnnxLoadFailure ()
hasMatchingShape node attribute dims = filterIntsAttribute attribute (== map fromIntegral dims) node

hasCorrectPadding :: (KnownNat padLeft, KnownNat padRight, KnownNat padTop, KnownNat padBottom)
                  => P.NodeProto -> Proxy padLeft -> Proxy padRight -> Proxy padTop -> Proxy padBottom -> Either OnnxLoadFailure ()
hasCorrectPadding node ppl ppr ppt ppb 
  = let left   = fromIntegral $ natVal ppl
        right  = fromIntegral $ natVal ppr
        top    = fromIntegral $ natVal ppt
        bottom = fromIntegral $ natVal ppb
        paddings = case readIntsAttribute "pads" node of
                     Right ps -> ps
                     Left  _  -> [0, 0, 0, 0]
     in case paddings of
          [left', top', right', bottom'] -> guardOnnx "Padding dimensions mismatch" (left == left' && top == top' && right == right' && bottom == bottom')
          _                              -> loadFailureReason "Incorrect padding dimensions"


hasMatchingDouble :: KnownSymbol a => P.NodeProto -> Proxy a -> T.Text -> Either OnnxLoadFailure ()
hasMatchingDouble node a x = do 
  let a' = (read $ symbolVal a) :: Double
  x' <- readDoubleAttribute x node
  guardOnnx ("Value mismatch for attribute " ++ T.unpack x ++ " of type double") (x' == a')

hasMatchingInt :: KnownNat a => P.NodeProto -> Proxy a -> T.Text -> Either OnnxLoadFailure ()
hasMatchingInt node a x = do
  let a' = fromIntegral $ natVal a
  x' <- readIntAttribute x node
  guardOnnx ("Value mismatch for attribute " ++ T.unpack x ++ " of type int") (x' == a')


onnxIncorrectNumberOfInputs :: Either OnnxLoadFailure a
onnxIncorrectNumberOfInputs = loadFailureReason "Incorrect number of inputs"
