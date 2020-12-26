{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}

import           Numeric.LinearAlgebra

import           Grenade
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Pooling

import           Criterion.Main

main :: IO ()
main = do
  putStrLn $ "Benchmarking with type: " ++ nameF
  x :: S ('D2 60 60) <- randomOfShape
  y :: S ('D3 60 60 1) <- randomOfShape
  defaultMain
    [ bgroup
        "im2col"
        [ bench "im2col 28x28"     $ whnf (im2col 5 5 1 1) ((28 >< 28) [1 ..])
        , bench "im2col 100x100"   $ whnf (im2col 10 10 1 1) ((100 >< 100) [1 ..])
        , bench "im2col 3x416x416" $ whnf (vid2col 3 3 1 1 416 416) (((3 * 416) >< 416) [1 ..])
        ]
    , bgroup
        "col2im"
        [ bench "col2im 28x28"     $ whnf (col2im 5 5 1 1 28 28) ((576 >< 25) [1 ..])
        , bench "col2im 100x100"   $ whnf (col2im 10 10 1 1 100 100) ((8281 >< 100) [1 ..])
        , bench "col2im 3x416x416" $ whnf (col2vid 1 1 1 1 414 414) (((414 * 414) >< 27) [1 ..])
        ]
    , bgroup
        "poolfw"
        [ bench "poolforwards 3x4" $ whnf (poolForward 1 3 4 2 2 1 1) ((3 >< 4) [1 ..])
        , bench "poolforwards 28x28" $ whnf (poolForward 1 28 28 5 5 1 1) ((28 >< 28) [1 ..])
        , bench "poolforwards 100x100" $ whnf (poolForward 1 100 100 10 10 1 1) ((100 >< 100) [1 ..])
        ]
    , bgroup
        "poolbw"
        [ bench "poolbackwards 3x4" $ whnf (poolBackward 1 3 4 2 2 1 1 ((3 >< 4) [1 ..])) ((2 >< 3) [1 ..])
        , bench "poolbackwards 28x28" $ whnf (poolBackward 1 28 28 5 5 1 1 ((28 >< 28) [1 ..])) ((24 >< 24) [1 ..])
        , bench "poolbackwards 100x100" $ whnf (poolBackward 1 100 100 10 10 1 1 ((100 >< 100) [1 ..])) ((91 >< 91) [1 ..])
        ]
    , bgroup
        "padcrop"
        [ bench "pad 2D 60x60" $ nf (testRun2D Pad) x
        , bench "pad 3D 60x60" $ nf (testRun3D Pad) y
        , bench "crop 2D 60x60" $ nf (testRun2D' Crop) x
        , bench "crop 3D 60x60" $ nf (testRun3D' Crop) y
        ]
    ]
  putStrLn $ "Benchmarked with type: " ++ nameF

testRun2D :: Pad 1 1 1 1 -> S ('D2 60 60) -> S ('D2 62 62)
testRun2D = snd ... runForwards

testRun3D :: Pad 1 1 1 1 -> S ('D3 60 60 1) -> S ('D3 62 62 1)
testRun3D = snd ... runForwards

testRun2D' :: Crop 1 1 1 1 -> S ('D2 60 60) -> S ('D2 58 58)
testRun2D' = snd ... runForwards

testRun3D' :: Crop 1 1 1 1 -> S ('D3 60 60 1) -> S ('D3 58 58 1)
testRun3D' = snd ... runForwards

(...) :: (a -> b) -> (c -> d -> a) -> c -> d -> b
(...) = (.) . (.)
