{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE CPP                       #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}

import           Control.Applicative
import           Control.Concurrent
import           Control.Monad.Random

import           Options.Applicative

import           Graphics.Image               hiding (map, on)

import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Static as H

import           Grenade


data SuperResOptions = SuperResOptions FilePath         -- onnx file
                                       FilePath         -- input image

pSuperRes :: Parser SuperResOptions
pSuperRes = SuperResOptions <$> argument str (metavar "onnx") <*> argument str (metavar "image")

loadSuperResImage :: FilePath -> IO (Maybe (S ('D3 224 224 1), [[RealNum]], [[RealNum]]))
loadSuperResImage path = do
  img <- readImageRGB VU path
  -- displayImage img
  return $ do
    guard $ dims img == (224, 224)
    let imgYCbCr = toImageYCbCr img
        imgY0    = map (\(PixelYCbCr y _ _ )      -> doubleToRealNum y )  . concat . toLists $ imgYCbCr
        imgCb    = map (map (\(PixelYCbCr _ cb _) -> doubleToRealNum cb)) . toLists $ imgYCbCr
        imgCr    = map (map (\(PixelYCbCr _ _ cr) -> doubleToRealNum cr)) . toLists $ imgYCbCr

    return (S3D (H.fromList imgY0), imgCb, imgCr)

displayHighResImage :: S ('D3 672 672 1) -> [[RealNum]] -> [[RealNum]] -> IO ()
displayHighResImage (S3D m) cbs crs = do
  let realNumToPixel = PixelX . realNumToDouble
      m'  = LA.toLists $ H.extract m              :: [[RealNum]]
      m'' = map (map realNumToPixel) m'           :: [[Pixel X Double]]
      img = fromLists m''                         :: Image VU X Double

      imgBs  = fromLists $ map (map realNumToPixel) cbs :: Image VU X Double
      imgRs  = fromLists $ map (map realNumToPixel) crs :: Image VU X Double

      imgBs' = resize Bilinear Edge (672, 672) imgBs :: Image VU X Double
      imgRs' = resize Bilinear Edge (672, 672) imgRs :: Image VU X Double

      finalImg = fromImagesX [(LumaYCbCr, img), (CBlueYCbCr, imgBs'), (CRedYCbCr, imgRs')] :: Image VU YCbCr Double

  displayImage finalImg
  threadDelay 10000000

main :: IO ()
main = do
    SuperResOptions netPath imgPath <- execParser (info (pSuperRes <**> helper) idm)
    res <- loadSuperResolution netPath

    inputM <- loadSuperResImage imgPath

    case (res, inputM) of
      (Right net, Just (input, cbs, crs))  -> do
        let S3D y = runNet net input
        displayHighResImage (S3D y) cbs crs

      (Left err, _) -> print err
      _             -> putStrLn "Failed to load network and file"
