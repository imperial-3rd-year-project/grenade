{-# LANGUAGE CPP                       #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}

import           Control.Applicative
import           Control.Monad
import           Graphics.Image               hiding (map, on)
import           Numeric.LinearAlgebra.Static (L, fromList, (===))
import           Options.Applicative

import           Grenade
import           Grenade.Utils.PascalVoc

data TinyYoloV2Options = TinyYoloV2Options FilePath  -- onnx file
                                           FilePath  -- image path

pTinyYoloV2 :: Parser TinyYoloV2Options
pTinyYoloV2 = TinyYoloV2Options <$> argument str (metavar "onnx") <*> argument str (metavar "image")

loadTinyYoloImage :: FilePath -> IO (Maybe (S ('D3 416 416 3)))
loadTinyYoloImage path = do
  img <- readImageRGB VU path
  displayImage img
  return $ do
    guard $ dims img == (416, 416)
    let [reds, greens, blues] = map (fromList . map (\(PixelX y) -> y) . concat . toLists)
                              $ toImagesX img :: [L 416 416]
    return $ S3D $ reds === greens === blues

main :: IO ()
main = do
    TinyYoloV2Options netPath imgPath <- execParser (info (pTinyYoloV2 <**> helper) idm)
    res  <- loadTinyYoloV2 netPath
    imgM <- loadTinyYoloImage imgPath

    case (res, imgM) of
      (Right net, Just img) -> do
        putStrLn "Succesfully loaded image and network"
        print $ processOutput (runNet net img) 0.3

      (Left err, _) -> print err
