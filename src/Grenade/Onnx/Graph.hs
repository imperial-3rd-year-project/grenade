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
{-# LANGUAGE StandaloneDeriving  #-}
{-# LANGUAGE TupleSections       #-}
{-|
Module      : Grenade.Onnx.Graph
Description : Data type representing Onnx graphs along with functions to construct these.
-}

module Grenade.Onnx.Graph
  ( Composition (..)
  , SPG (..)
  , generateGraph
  , generateInitializerMap
  )
  where

import           Control.Applicative          ((<|>))
import           Data.Bifunctor               (bimap)
import           Data.Either                  (partitionEithers)
import           Data.List                    (foldl')
import qualified Data.Map.Strict              as Map
import           Data.ProtoLens.Labels        ()
import qualified Data.Text                    as T

import           Grenade.Onnx.OnnxLoadFailure
import           Grenade.Onnx.Utils

import           Lens.Micro
import qualified Proto.Onnx                   as P


data Composition = S | P

-- | Data type representing series-parallel graphs.
--
--   Using the DataKinds extension, we can specify whether a constructor is
--   composing graphs in series or in parallel, hence the term construction
--   is verified by the type system to be correct.
data SPG (s :: Composition) a where
  Node     :: a -> SPG s a
  Series   :: [SPG 'P a] -> SPG 'S a
  Parallel :: [SPG 'S a] -> SPG 'P a

deriving instance Functor (SPG s)

graphCons :: a -> SPG s a -> SPG 'S a
graphCons x (Node x') = Series [Node x, Node x']
graphCons x (Series xs) = Series (Node x : xs)
graphCons x x'@(Parallel _) = Series [Node x, x']

wrapSeries :: SPG s a -> SPG 'S a
wrapSeries (Node x) = Node x
wrapSeries x@(Parallel _) = Series [x]
wrapSeries xs@(Series _) = xs

graphAppend :: SPG s a -> SPG s' a -> SPG 'S a
graphAppend (Node x) graph = x `graphCons` graph
graphAppend (Series xs) (Series xs') = Series (xs ++ xs')
graphAppend (Series xs) (Node x) = Series (xs ++ [Node x])
graphAppend xs ys = wrapSeries xs `graphAppend` wrapSeries ys

graphHead :: SPG s a -> SPG s a
graphHead (Node x) = Node x
graphHead (Series xs) = case xs of
                          (x : _) -> Series [graphHead x]
                          _       -> error "Graph is empty"
graphHead (Parallel xs) = Parallel (graphHead <$> xs)

generateInitializerMap :: P.GraphProto -> Map.Map T.Text P.TensorProto
generateInitializerMap graph = foldl' (\map node -> Map.insert (node ^. #name) node map) Map.empty
                             $ graph ^. #initializer

generateGraph :: P.ModelProto -> Either OnnxLoadFailure (P.GraphProto, SPG 'S P.NodeProto)
generateGraph model = case graphProto ^. #node of
                        nodes@(node:_) -> (graphProto, ) <$> generateGraph' node nodes
                        _              -> loadFailureReason "Model is empty"
  where
    graphProto     = model ^. #graph

generateGraph' :: P.NodeProto -> [P.NodeProto] -> Either OnnxLoadFailure (SPG 'S P.NodeProto)
generateGraph' node nodes = fst <$> genGraph node
  where
    (inputNodes, outputNodes) = foldl' classifyNode (Map.empty, Map.empty) nodes

    classifyNode :: (Map.Map T.Text [P.NodeProto], Map.Map T.Text [P.NodeProto])
                 -> P.NodeProto
                 -> (Map.Map T.Text [P.NodeProto], Map.Map T.Text [P.NodeProto])
    classifyNode (inputNodes, outputNodes) node
      | node ^. #opType == "Constant" = (inputNodes, outputNodes)
      | otherwise = (updateMap inputNodes input, updateMap outputNodes output)
      where
        input = node ^. #input
        output = node ^. #output

        updateMap = foldl' (flip insert)

        insert = Map.alter (Just . (\case
          Just xs -> node : xs
          Nothing -> [node]))


    genGraph :: P.NodeProto -> Either OnnxLoadFailure (SPG 'S P.NodeProto, Maybe P.NodeProto)
    genGraph node = case inputs of
                      (_ : _ : _) -> Right (Series [], Just node)
                      _           -> genGraphNode node
      where
        findNodes nodes = concatMap (\name -> Map.findWithDefault [] name nodes)

        inputNames = node ^. #input
        inputs = findNodes outputNodes inputNames

        genGraphNode :: P.NodeProto -> Either OnnxLoadFailure (SPG 'S P.NodeProto, Maybe P.NodeProto)
        genGraphNode node = bimap (over lastSuccessfulNode (<|> Just node)) (over _1 (node `graphCons`)) (genGraph' outputs)
          where
            outputNames = node ^. #output
            outputs = findNodes inputNodes outputNames

            genGraph' :: [P.NodeProto] -> Either OnnxLoadFailure (SPG 'S P.NodeProto, Maybe P.NodeProto)
            genGraph' []  = Right (Series [], Nothing)

            genGraph' [x] = genGraph x


            genGraph' xs = case partitionEithers (map genGraph xs) of
                             ([], rs) -> case unzip rs of
                               (parGraphs, Just next : _) -> over _1 (Parallel parGraphs `graphAppend`) <$> genGraphNode next
                               (_        , Nothing   : _) -> loadFailure "Missing combine node" (Just node) Nothing []
                               (_        , []           ) -> error "Unexpected empty parallel layer"
                             (err : _, _) -> Left err

