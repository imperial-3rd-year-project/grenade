{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Grenade.Onnx.Onnx where

import Data.ProtoLens.Labels ()
import qualified Proto.Onnx as P
import Lens.Micro
import qualified Data.Map.Strict as Map
import Data.Foldable (foldl')
import qualified Data.Text as T
import qualified Data.ByteString as B
import Data.ProtoLens.Encoding (decodeMessage)

data Composition = S | P

data SPG (s :: Composition) a where
  Node     :: a -> SPG s a
  Series   :: [SPG 'P a] -> SPG 'S a
  Parallel :: [SPG 'S a] -> SPG 'P a

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

generateInitializerMap :: P.GraphProto -> Map.Map T.Text P.TensorProto
generateInitializerMap graph = foldl' (\map node -> Map.insert (node ^. #name) node map) Map.empty
                             $ graph ^. #initializer

generateGraph :: P.ModelProto -> (P.GraphProto, SPG 'S P.NodeProto)
generateGraph model = (graphProto, graph)
  where
    graphProto     = model ^. #graph
    nodes@(node:_) = graphProto ^. #node
    (graph, _)     = genGraph node

    (inputNodes, outputNodes) = foldl' classifyNode (Map.empty, Map.empty) nodes

    classifyNode :: (Map.Map T.Text [P.NodeProto], Map.Map T.Text [P.NodeProto])
                 -> P.NodeProto
                 -> (Map.Map T.Text [P.NodeProto], Map.Map T.Text [P.NodeProto])
    classifyNode (inputNodes, outputNodes) node =
      (updateMap inputNodes input, updateMap outputNodes output)
      where
        input = node ^. #input
        output = node ^. #output

        updateMap = foldl' (\m k -> insert k m)

        insert = Map.alter (Just . (\case
          Just xs -> node : xs
          Nothing -> [node]))


    genGraph :: P.NodeProto -> (SPG 'S P.NodeProto, Maybe P.NodeProto)
    genGraph node = case inputs of
                      (_ : _ : _) -> (Series [], Just node)
                      _           -> genGraphNode node
      where
        findNodes nodes = concatMap (\name -> Map.findWithDefault [] name nodes)

        inputNames = node ^. #input
        inputs = findNodes outputNodes inputNames

        genGraphNode :: P.NodeProto -> (SPG 'S P.NodeProto, Maybe P.NodeProto)
        genGraphNode node = let (graph, next) = genGraph' outputs
                             in (node `graphCons` graph, next)
          where
            outputNames = node ^. #output
            outputs = findNodes inputNodes outputNames

            genGraph' :: [P.NodeProto] -> (SPG 'S P.NodeProto, Maybe P.NodeProto)
            genGraph' []  = (Series [], Nothing)

            genGraph' [x] = genGraph x

            genGraph' xs  = (Parallel parGraphs `graphAppend` remGraph, next')
              where 
                (parGraphs, Just next : _) = unzip (map genGraph xs)
                (remGraph, next') = genGraphNode next

readOnnxModel :: FilePath -> IO (Maybe P.ModelProto)
readOnnxModel path = do
  modelContent <- B.readFile path
  let (model :: Either String P.ModelProto) = decodeMessage modelContent
  case model of
    Left err -> error err
    Right modelProto -> return (Just $ modelProto)
