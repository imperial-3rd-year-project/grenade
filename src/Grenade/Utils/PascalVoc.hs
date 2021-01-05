{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs     #-}
{-|
Module      : Grenade.Utils.PascalVoc
Description : TODO Theo what is this for
-}

module Grenade.Utils.PascalVoc where

import           Grenade.Core.Shape

import           Data.List                    (elemIndices)
import qualified Numeric.LinearAlgebra.Data   as NLA
import qualified Numeric.LinearAlgebra.Static as H

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

-- | (x, y, w, h, confidence, label)
type DetectedObject
  = ( Int -- x
    , Int -- y
    , Int -- w
    , Int -- h
    , Double -- confidence
    , String -- label
    )

-- | Given the output of TinyYoloV2, finds all bounding boxes
--   with confidence higher than a threshold, and their labels
processOutput :: S ('D3 13 13 125) -> Double -> [DetectedObject]
processOutput (S3D mat) threshold = map toDetectedObject filtered
  where
    filtered = filter (\(_, _, _, _, c, probs) -> snd (argMax probs) * c > threshold ) boxDescs
    out      = H.extract mat
    boxDescs = [ getBoundingBoxDesc out i j box | i <- [0..12], j <- [0..12], box <- [0..4]]
    
    toDetectedObject (x, y, w, h, c, probs) = (l, r, t, b, c, labels !! (fst $ argMax probs))
      where
        l = floor $ x - w / 2
        r = floor $ x + w / 2
        t = floor $ y - h / 2
        b = floor $ y + h / 2
    
    -- tuple of (idx, max elem)
    argMax probs = (i, m)
      where
        m = maximum probs
        i = head $ elemIndices m probs
    

-- | TODO Theo
getDetails :: NLA.Matrix Double -> Int -> Int -> Int -> (Double, Double, Double, Double, Double)
getDetails out cy cx b = (tx, ty, tw, th, tc)
  where
    channel = b * 25
    tx = NLA.atIndex out (channel       * 13 + cy, cx)   -- out NLA.! offset 0 NLA.! j
    ty = NLA.atIndex out ((channel + 1) * 13 + cy, cx)  --out NLA.! offset 1 NLA.! j
    tw = NLA.atIndex out ((channel + 2) * 13 + cy, cx)  --out NLA.! offset 2 NLA.! j
    th = NLA.atIndex out ((channel + 3) * 13 + cy, cx)  --out NLA.! offset 3 NLA.! j
 
    tc = NLA.atIndex out ((channel + 4) * 13 + cy, cx)  --out NLA.! offset 4 NLA.! j

-- | TODO Theo
getProbs :: NLA.Matrix Double -> Int -> Int -> Int -> [Double]
getProbs out cy cx b = probs
  where
    channel = b * 25
    probs = [NLA.atIndex out ((channel + c) * 13 + cy, cx) | c <- [5..24] ]



-- | Given a cell, 0 <= i, j <= 12, and a bounding box 0 <= box <= 4, gives
--   the x, y, width, height for the bounding box, the confidence score,
--   and the probability distribution over the 20 available classes
getBoundingBoxDesc :: NLA.Matrix Double -> Int -> Int -> Int -> (Double, Double, Double, Double, Double, [Double])
getBoundingBoxDesc out cy cx b = (x, y, w, h, confidence * (maximum classes), classes)
  where
    anchors    = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    numClasses = 20
    channel    = b * 25

    tx = NLA.atIndex out ((channel + 0) * 13 + cy, cx)  -- out NLA.! offset 0 NLA.! j
    ty = NLA.atIndex out ((channel + 1) * 13 + cy, cx)  --out NLA.! offset 1 NLA.! j
    tw = NLA.atIndex out ((channel + 2) * 13 + cy, cx)  --out NLA.! offset 2 NLA.! j
    th = NLA.atIndex out ((channel + 3) * 13 + cy, cx)  --out NLA.! offset 3 NLA.! j
    tc = NLA.atIndex out ((channel + 4) * 13 + cy, cx)  --out NLA.! offset 4 NLA.! j

    x = (fromIntegral cx + sigmoid tx) * 32
    y = (fromIntegral cy + sigmoid ty) * 32

    w = exp tw * (anchors !! (2 * b    )) * 32
    h = exp th * (anchors !! (2 * b + 1)) * 32

    confidence = sigmoid tc

    classes = softmax [ NLA.atIndex out ((channel + 5 + c) * 13 + cy, cx) | c <- [0..numClasses - 1] ]  

    sigmoid :: Double -> Double
    sigmoid = (1 /) . (1 +) . exp . negate

    softmax :: [Double] -> [Double]
    softmax xs = map (/sumxs) expxs
      where
        expxs = map exp xs
        sumxs = sum expxs
