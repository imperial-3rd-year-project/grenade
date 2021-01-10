module Test.Grenade.Onnx.Utils where

import qualified Data.Text            as T

import Grenade

import           Hedgehog
import           Hedgehog.Gen           as Gen
import           Hedgehog.Range         as Range

import           Data.ProtoLens.Message    (Message)
import           Data.ProtoLens.Encoding   (encodeMessage, decodeMessage)
import           Proto.Onnx             as P

import           Lens.Micro

edgeNames :: [T.Text]
edgeNames = T.pack . ("Edge #" ++) . show <$> [(1 :: Int) ..]

edge1, edge2, edge3, edge4 :: T.Text
[edge1, edge2, edge3, edge4] = take 4 edgeNames

castMessage :: (Message from, Message to) => from -> Either String to
castMessage = decodeMessage . encodeMessage

elementRemove :: MonadGen m => [a] -> m (a, [a])
elementRemove [] = error "Used with empty list"
elementRemove xs = do
  ix <- Gen.int $ Range.constant 0 (length xs - 1)
  let (xs', (x : xs'')) = splitAt ix xs
  pure (x, xs' ++ xs'')

randomMerge :: [[a]] -> Gen [a]
randomMerge xss = merge' (Prelude.filter (not . null) xss)
  where
    merge' [] = return []
    merge' xss' = do
      ((x : xs), xss'') <- elementRemove xss'
      (x :) <$> randomMerge (xs : xss'')

loadModel :: OnnxLoadable a => P.ModelProto -> Either OnnxLoadFailure a
loadModel = loadOnnxModel' . encodeMessage
