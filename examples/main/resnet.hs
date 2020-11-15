{-# LANGUAGE CPP                       #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}

import           Control.Applicative
import           Control.Monad.Random
import           Control.Monad.Trans.Except

import qualified Data.Attoparsec.Text         as A
import           Data.Function                (on)
import           Data.List
import qualified Data.Text                    as T
import qualified Data.Text.IO                 as T
import qualified Data.Vector.Storable         as V
import Debug.Trace
import           Options.Applicative

import           Graphics.Image               hiding (map, on)
import           Graphics.Image.Interface     (toDouble)
import           Graphics.Image.IO

import qualified Numeric.LinearAlgebra        as LA
import           Numeric.LinearAlgebra.Static (L, R)
import qualified Numeric.LinearAlgebra.Static as H


import           Grenade
import           Grenade.Networks.ResNet18


data ResNetOptions = ResNetOptions FilePath         -- onnx file
                                   FilePath         -- input image

pResnet :: Parser ResNetOptions
pResnet = ResNetOptions <$> argument str (metavar "onnx") <*> argument str (metavar "image")

-- loadResNetImage :: FilePath -> IO (Maybe (S ('D3 224 224 3)))
loadResNetImage :: FilePath -> IO (Maybe (S ('D3 224 224 3)))
loadResNetImage path = do
  img <- readImageRGB VU path
  displayImage img
  return $ do
    guard $ dims img == (224, 224)
    let [img_red, img_green, img_blue] = toImagesX img
        [reds, greens, blues]          = map (map (\(PixelX y) -> y) . concat . toLists) $ [img_red, img_green, img_blue]

        redM   = H.dmmap (\a -> (a - 0.485) / 0.229) $ H.fromList reds   :: L 224 224
        greenM = H.dmmap (\a -> (a - 0.456) / 0.224) $ H.fromList greens :: L 224 224
        blueM  = H.dmmap (\a -> (a - 0.406) / 0.225) $ H.fromList blues  :: L 224 224

        mat    = redM H.=== greenM H.=== blueM

    return (S3D mat)

main :: IO ()
main = do
    ResNetOptions netPath imgPath <- execParser (info (pResnet <**> helper) idm)
    res <- loadResNet netPath

    inputM <- loadResNetImage imgPath

    case (res, inputM) of
      (Just net, Just input)  -> do
        let S1D y = runNet net input
            tops  = getTop 5 $ LA.toList $ H.extract y
        putStrLn $ show tops
      _ -> putStrLn "fail"
  where
    getTop :: Ord a => Int -> [a] -> [Int]
    getTop n xs = map fst $ take n $ sortBy (compare `on` snd) $ zip [0..] xs

