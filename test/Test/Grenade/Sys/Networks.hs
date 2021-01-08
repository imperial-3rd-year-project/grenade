{-# LANGUAGE CPP              #-}
{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-# LANGUAGE TemplateHaskell  #-}
{-# LANGUAGE TypeOperators    #-}

module Test.Grenade.Sys.Networks where

import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Data   as D
import           Numeric.LinearAlgebra.Static (L)
import qualified Numeric.LinearAlgebra.Static as H

import           Hedgehog

import           Test.Grenade.Sys.Utils
import           Test.Hedgehog.Compat

import           Data.List
import           Data.Maybe
import           Data.Function
import           Data.Either
import           System.FilePath

import           Grenade
import           Grenade.Utils.PascalVoc

-- Checks that the differences in the reference and the output from the network are "acceptable". Due to floating
-- point errors, using doubles instead of floats, etc. the outputs can be not precisely the same. We check that
-- each box given by the reference output has a corresponding box in ours which roughly overlaps with it, detects
-- the same object, with approximately the same confidence, and vice versa for our output compared to the reference.
yoloAcceptableDifference :: [(Int, Int, Int, Int, Double, String)] -> [(Int, Int, Int, Int, Double, String)] -> Bool
yoloAcceptableDifference ref out = all (isJust . (searchForMatch out)) ref && all (isJust . (searchForMatch ref)) out
  where    
    searchForMatch searchSpace (li, ri, ti, bi, p, x)
      | null matches = Nothing
      | otherwise    = Just $ head matches
      where
        -- Find all entries with the same name
        sameName = filter (\(_, _, _, _, _, x') -> x == x') searchSpace
        -- Find entries with sufficient overlap
        overlap = \(li', ri', ti', bi', _, _) -> sufficientOverlap (li, ri, ti, bi) (li', ri', ti', bi')
        overlapping = filter overlap sameName
        -- Finally, ensure that the probabilities are within a threshold
        matches = filter (\(_, _, _, _, p', _) -> abs p - p' <= 0.1) overlapping

    sufficientOverlap (li, ri, ti, bi) (li', ri', ti', bi') = overlap >= 0.6
      where
        (l, r, t, b) = (fromIntegral li, fromIntegral ri, fromIntegral ti, fromIntegral bi)
        (l', r', t', b') = (fromIntegral li', fromIntegral ri', fromIntegral ti', fromIntegral bi')
        xOverlap = maximum [ 0, minimum [r, r'] - maximum [l, l'] ]
        yOverlap = maximum [ 0, minimum [b, b'] - maximum [t, t'] ]
        overlap1 = (xOverlap * yOverlap) / ((r - l) * (b - t))
        overlap2 = (xOverlap * yOverlap) / ((r' - l') * (b' - t'))
        -- Get the least overlap, to avoid the case where a very large/small
        -- and incorrect bounding box output is treated as being correct
        overlap = minimum [overlap1, overlap2] :: Double

-- Test that the bundled TinyYOLOv2 network, stored as an ONNX file
-- loads correctly and gives expected output when given a test image
prop_yolo_loads_and_finds_objects :: Property 
prop_yolo_loads_and_finds_objects = withTests 1 $ property $ do
  img <- evalIO (loadSerializedImage (imagesDir </> "person") :: IO (Maybe (S ('D3 416 416 3))))
  netPath <- evalIO $ getPathForNetwork TinyYoloV2
  net     <- evalIO $ loadTinyYoloV2 netPath

  assert $ isJust img
  assert $ isRight net

  let Just img'  = img
      Right net' = net
      probs = processOutput (runNet net' img') 0.3
      -- Values taken from running the TinyYolov2 network bundled
      -- with grenade through the ONNX inference runtime Python
      -- module
      reference = [ (75, 163, 97, 371,   0.7742048036087853, "person")
                  , (2, 86, 262, 354,    0.7187669842556927, "sheep" )
                  , (283, 409, 151, 335, 0.6768828930128936, "sheep" )
                  , (305, 410, 155, 323, 0.5562674827033209, "sheep" ) ]
  assert $ yoloAcceptableDifference reference probs

-- Test that the bundled ResNet-18 network, stored as an ONNX file,
-- loads correctly and gives expected output when given a test image
prop_resnet_loads_and_finds_objects :: Property 
prop_resnet_loads_and_finds_objects = withTests 1 $ property $ do
  img <- evalIO (loadSerializedImage (imagesDir </> "dog") :: IO (Maybe (S ('D3 224 224 3))))
  netPath <- evalIO $ getPathForNetwork ResNet18
  net     <- evalIO $ loadResNet netPath

  assert $ isJust img
  assert $ isRight net

  let Just img'  = img
      Right net' = net
      S1D y = runNet net' img'
      tops  = getTop 5 $ LA.toList $ H.extract y
  tops === [162,166,167,215,164] -- beagle, Walker hound, English foxhound, Brittany spaniel, bluetick
  where
    getTop :: Ord a => Int -> [a] -> [Int]
    getTop n xs = map fst $ take n $ sortBy (flip compare `on` snd) $ zip [0..] xs

prop_superresolution_loads_and_upsamples :: Property
prop_superresolution_loads_and_upsamples = withTests 1 $ property $ do
  imgY0 <- evalIO (loadSerializedChannel (imagesDir </> "cheetah_Y0") :: IO (Maybe (S ('D3 224 224 1))))
  netPath <- evalIO $ getPathForNetwork SuperResolution
  net     <- evalIO $ loadSuperResolution netPath

  assert $ isJust imgY0
  assert $ isRight net

  let Just input = imgY0
      Right net' = net
      (S3D input') = input
      (S3D v)      = runNet net' input :: S ('D3 672 672 1)
      down         = downsample v
      diffs        = zipWith (\x y -> abs (x - y)) (concat $ D.toLists $ H.extract down) (concat $ D.toLists $ H.extract input')
      avgDiff      = (sum diffs) / (224 * 224)
  isWithinOf 0.01 avgDiff 0
  where
    downsample :: L 672 672 -> L 224 224
    downsample mat = H.build builder
      where
        mat' = H.extract mat
        builder i j = (sum vals) / 9
          where
            ro = (floor i) * 3
            co = (floor j) * 3
            vals = [mat' `D.atIndex` (ro', co') | ro' <- [ro..ro + 2], co' <- [co..co + 2]]

tests :: IO Bool
tests = checkParallel $$(discover)
