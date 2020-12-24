{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs               #-}

import           Criterion.Main
import           Grenade

main :: IO ()
main = do
  (!inputYolo) :: S ('D3 416 416 3) <- randomOfShape
  yoloPath <- getPathForNetwork TinyYoloV2
  !yolo <- loadTinyYoloV2 yoloPath
  
  (!inputResnet) :: S ('D3 224 224 3) <- randomOfShape
  resnetPath <- getPathForNetwork ResNet18
  !resnet <- loadResNet resnetPath
  
  (!inputSuper) :: S ('D3 224 224 1) <- randomOfShape
  superPath <- getPathForNetwork SuperResolution
  !super <- loadSuperResolution superPath
  
  case (yolo, resnet, super) of
    (Right y, Right r, Right s) -> do
      defaultMain [bgroup "run"  [ bench "resnet" $ nf (runNet r) inputResnet
                                 , bench "yolo"   $ nf (runNet y) inputYolo
                                 , bench "super"  $ nf (runNet s) inputSuper]]
    (Left err, _, _) -> print $ "Error in yolo: " ++ (show err)
    (_, Left err, _) -> print $ "Error in resnet: " ++ (show err)
    (_, _, Left err) -> print $ "Error in super-res: " ++ (show err)