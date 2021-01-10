{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TupleSections     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE ViewPatterns      #-}
{-# LANGUAGE TypeFamilies      #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

module Test.Grenade.Onnx.Graph where

import Test.Grenade.Onnx.Utils

import           Data.List                 (foldl')
import qualified Data.Text              as T

import           Grenade.Onnx.Graph     as G

import           Hedgehog
import           Hedgehog.Gen           as Gen
import           Hedgehog.Range         as Range

import           Data.ProtoLens            (defMessage)
import           Data.ProtoLens.Labels     ()
import           Proto.Onnx             as P

import           Lens.Micro


-- | Generates a DAG that comforms to ONNX standards.
genDAG :: Gen [P.NodeProto]
genDAG = Gen.int (Range.linear 10 100) >>= genNodes edgeNames
  where
    genNodes :: [T.Text] -> Int -> Gen [P.NodeProto]
    genNodes names 0 = (\(_, node) -> [node]) <$> randomNode names
    genNodes outNames n = do
      (remNames, node) <- randomNode outNames
      nodes <- genNodes remNames (n - 1)
      numNodeOuts <- Gen.int (Range.linear 1 5)
      outEdges <- (head . (^. #input) <$>) . (take numNodeOuts) <$> Gen.shuffle nodes
      return ((node & #output %~ (outEdges ++)) : nodes)

    randomNode :: [T.Text] -> Gen ([T.Text], P.NodeProto)
    randomNode outNames = do
      numOuts <- Gen.int (Range.linear 1 5)
      let (outs, remNames) = splitAt numOuts outNames
      numIns  <- Gen.int (Range.linear 1 5)
      let (ins, remNames') = splitAt numIns remNames
      return (remNames', defMessage & #input .~ ins & #output .~ outs)

genSPG :: Gen [P.NodeProto]
genSPG = let (inName : outName : names) = edgeNames
          in snd <$> (Gen.int (Range.linear 5 10) >>= genSPG' inName outName names)
  where
    genSPG' :: T.Text -> T.Text -> [T.Text] -> Int -> Gen ([T.Text], [P.NodeProto])
    genSPG' inEdge outEdge names 0 =
      return (names, [defMessage & #input .~ [inEdge] & #output .~ [outEdge]])
    genSPG' inEdge outEdge names n =
      Gen.choice ((\f -> f inEdge outEdge names) <$> [genSeries, genParallel])
      where
        genSeries :: T.Text -> T.Text -> [T.Text] -> Gen ([T.Text], [P.NodeProto])
        genSeries inName outName (preOutName : postInName : names) = do
          (names', preNodes) <- genSPG' inName preOutName names (n - 1)
          (names'', postNodes) <- genSPG' postInName outName names' (n - 1)
          let node = defMessage & #input .~ [preOutName] & #output .~ [postInName]
          return (names'', preNodes ++ [node] ++ postNodes)
        genSeries _ _ _ = error "genSeries ran out of names"

        genParallel :: T.Text -> T.Text -> [T.Text] -> Gen ([T.Text], [P.NodeProto])
        genParallel inName outName names = do
          numBranches <- Gen.int (Range.linear 1 5)
          (remNames, branches) <- genPar numBranches names
          let (#input .~ [inName] -> inNode, #output .~ [outName] -> outNode, allBranches)
                = foldl' appendBranch (defMessage, defMessage, []) branches
          nodes <- randomMerge allBranches
          return (remNames, inNode : nodes ++ [outNode])
              
        appendBranch :: (P.NodeProto, P.NodeProto, [[P.NodeProto]])
                     -> (T.Text, T.Text, [P.NodeProto])
                     -> (P.NodeProto, P.NodeProto, [[P.NodeProto]])
        appendBranch (inNode, outNode, nss) (branchIn, branchOut, ns) =
          (append branchIn #output inNode, append branchOut #input outNode, ns : nss)
          where append br = (%~ (br :))

        genPar :: Int -> [T.Text] -> Gen ([T.Text], [(T.Text, T.Text, [P.NodeProto])])
        genPar 0 names = return (names, [])
        genPar n names = do
          (remNames, inName, outName, nodes) <- genBranch names
          (remNames', branches) <- genPar (n - 1) remNames
          return (remNames', (inName, outName, nodes) : branches)
          
        genBranch :: [T.Text] -> Gen ([T.Text], T.Text, T.Text, [P.NodeProto])
        genBranch (inName : outName : remNames) = do
          (remNames', nodes) <- genSPG' inName outName remNames (n - 1)
          return (remNames', inName, outName, nodes)
        genBranch _ = error "genBranch ran out of names"
      
genGraph :: Gen [P.NodeProto] -> Gen P.GraphProto
genGraph gen = (defMessage &) . (#node .~) <$> gen

genModel :: Gen [P.NodeProto] -> Gen P.ModelProto
genModel gen = (defMessage &) . (#graph .~) <$> genGraph gen

prop_generateGraph_does_not_crash = property $ do
  model <- forAll $ genModel genDAG
  grenadeModel <- evalEither (castMessage model)
  _ <- evalNF (generateGraph grenadeModel)
  success

countSPGNodes :: SPG s a -> Int
countSPGNodes (G.Node _) = 1
countSPGNodes (G.Series xs) = sum (countSPGNodes <$> xs)
countSPGNodes (G.Parallel ys) = sum (countSPGNodes <$> ys)

prop_generateGraph_parses_SPG = property $ do
  model <- forAll $ genModel genSPG
  grenadeModel <- evalEither (castMessage model)
  (_, spg) <- evalEither (generateGraph grenadeModel)
  let graphNodes = length $ model ^. #graph . #node
      spgNodes = countSPGNodes spg
  
  graphNodes === spgNodes

tests :: IO Bool
tests = checkParallel $$(discover)
