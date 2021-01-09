{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs     #-}
{-|
Module      : Grenade.Utils.PascalVoc
Description : Post-processing for TinyYoloV2
-}

module Grenade.Utils.PascalVoc (DetectedObject, labels, processOutput) where

import           Data.List                    (elemIndices)
import qualified Numeric.LinearAlgebra.Data   as NLA
import qualified Numeric.LinearAlgebra.Static as H

import           Grenade.Core.Shape
import           Grenade.Types

-- | List of possible labels of TinyYoloV2
labels :: [String]
labels
  = [ "aeroplane"
    , "bicycle"
    , "bird"
    , "boat"
    , "bottle"
    , "bus"
    , "car"
    , "cat"
    , "chair"
    , "cow"
    , "diningtable"
    , "dog"
    , "horse"
    , "motorbike"
    , "person"
    , "pottedplant"
    , "sheep"
    , "sofa"
    , "train"
    , "tvmonitor"
    ]

-- | Type alias for output from Yolo neural network.
--
--   Stores (x position, y position, width height, confidence score, object label).
type DetectedObject = (Int, Int, Int, Int, RealNum, String)

-- | Given the output of TinyYoloV2, finds all bounding boxes
--   with confidence higher than a threshold, and their labels
processOutput :: S ('D3 13 13 125) -- ^ Output tensor from TinyYoloV2
              -> RealNum           -- ^ Minimum confidence for bounding boxes, between 0 and 1. A good default is 0.3.
              -> RealNum           -- ^ Maximum IoU ratio to consider boxes not overlapping, between 0 and 1. A good default is 0.5.
              -> [DetectedObject]
processOutput (S3D mat) confThreshold iouThreshold = collapseBoxes iouThreshold $ map toDetectedObject filtered
  where
    filtered = filter (\(_, _, _, _, c, probs) -> snd (argMax probs) * c > confThreshold ) boxDescs
    out      = H.extract mat
    boxDescs = [ getBoundingBoxDesc out i j box | i <- [0..12], j <- [0..12], box <- [0..4]]

    toDetectedObject (x, y, w, h, c, probs) = (l, r, t, b, c, labels !! (fst $ argMax probs))
      where
        l = floor $ x - w / 2
        r = floor $ x + w / 2
        t = floor $ y - h / 2
        b = floor $ y + h / 2

    -- Returns (index of maximum element, maximum element)
    argMax probs = (i, m)
      where
        m = maximum probs
        i = head $ elemIndices m probs
    
-- | Given an IoU threshold and a list of boxes, find overlapping boxes and collapse them into one.
--
--   Boxes are determined to be overlapping if their intersection over union value passes the threshold.
--   When boxes are merged we pick the box with higher confidence score.
--
--   IoU threshold should be between 0 and 1, we've found 0.5 to work quite well.
collapseBoxes :: RealNum -> [DetectedObject] -> [DetectedObject]
collapseBoxes threshold xs = filter (\y -> all (doesNotOverlapWith y) xs) xs
  where
    -- Consider objects as overlapping if they have same label and their intersection over union area is too big.
    -- Only consider overlapping the object which has lower confidence.
    doesNotOverlapWith :: DetectedObject -> DetectedObject -> Bool
    doesNotOverlapWith y@(left, right, top, bottom, conf, label) x@(left', right', top', bottom', conf', label')
      = label /= label' || y == x || iou < threshold || conf > conf'
        where
          ix1   = max left left'
          ix2   = min right right'
          iy1   = max top top'
          iy2   = min bottom bottom'
          area  = (right  - left)  * (bottom  - top)
          area' = (right' - left') * (bottom' - top')
          inter = if ix1 < ix2 && iy1 < iy2 then (ix2 - ix1) * (iy2 - iy1) else 0
          iou   = fromIntegral inter / fromIntegral (area + area' - inter)

-- | Given the output matrix from YOLO, a cell position cy, cx and a bounding box b,
--   outputs (x, y, width, height, confidence score, probability distribution over 20 classes)
--   for said box.
--
--   Must have that 0 <= cy, cx <= 12 and 0 <= b <= 4.
getBoundingBoxDesc :: NLA.Matrix RealNum -> Int -> Int -> Int -> (RealNum, RealNum, RealNum, RealNum, RealNum, [RealNum])
getBoundingBoxDesc out cy cx b = (x, y, w, h, confidence * (maximum classes), classes)
  where
    anchors    = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    numClasses = 20
    channel    = b * 25

    tx = NLA.atIndex out ((channel + 0) * 13 + cy, cx)
    ty = NLA.atIndex out ((channel + 1) * 13 + cy, cx)
    tw = NLA.atIndex out ((channel + 2) * 13 + cy, cx)
    th = NLA.atIndex out ((channel + 3) * 13 + cy, cx)
    tc = NLA.atIndex out ((channel + 4) * 13 + cy, cx)

    x = (fromIntegral cx + sigmoid tx) * 32
    y = (fromIntegral cy + sigmoid ty) * 32

    w = exp tw * (anchors !! (2 * b    )) * 32
    h = exp th * (anchors !! (2 * b + 1)) * 32

    confidence = sigmoid tc

    classes = softmax [ NLA.atIndex out ((channel + 5 + c) * 13 + cy, cx) | c <- [0..numClasses - 1] ]

    sigmoid :: RealNum -> RealNum
    sigmoid = (1 /) . (1 +) . exp . negate

    softmax :: [RealNum] -> [RealNum]
    softmax xs = map (/sumxs) expxs
      where
        expxs = map exp xs
        sumxs = sum expxs
