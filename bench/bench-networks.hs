{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}

import           Criterion.Main

import           Grenade

main :: IO ()
main = defaultMain
  [ env resnetEnv     $ \ ~(net, x) -> bench "Resnet18"        $ nf (runNet net) x
  , env tinyYoloV2Env $ \ ~(net, x) -> bench "TinyYoloV2"      $ nf (runNet net) x
  , env superResEnv   $ \ ~(net, x) -> bench "SuperResoltuion" $ nf (runNet net) x
  ]

tinyYoloV2Env :: IO (TinyYoloV2, S ('D3 416 416 3))
tinyYoloV2Env = do
  x          <- randomOfShape
  yoloPath   <- getPathForNetwork TinyYoloV2
  Right yolo <- loadTinyYoloV2 yoloPath
  return (yolo, x)

resnetEnv :: IO (ResNet18, S ('D3 224 224 3))
resnetEnv = do
  x            <- randomOfShape
  resnetPath   <- getPathForNetwork ResNet18
  Right resnet <- loadResNet resnetPath
  return (resnet, x)

superResEnv :: IO (SuperResolution, S ('D3 224 224 1))
superResEnv = do
  x           <- randomOfShape
  superPath   <- getPathForNetwork SuperResolution
  Right super <- loadSuperResolution superPath
  return (super, x)
