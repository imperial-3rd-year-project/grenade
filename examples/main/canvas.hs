{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE OverloadedStrings   #-}


import           Control.Applicative
import           Control.DeepSeq
import           Control.Monad
import           Control.Monad.Random
import           Control.Monad.Trans.Except

import           Data.Serialize
import qualified Data.ByteString              as B
import qualified Data.Attoparsec.Text         as A
import           Data.List                    (foldl', maximumBy)
import qualified Data.Text                    as T
import qualified Data.Text.IO                 as T
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Unboxed          as UB
import qualified Data.Vector                  as DV

import           Data.Ord
import           Data.Maybe                   (fromMaybe)
import           Data.Word
import           Data.Bits
import           Data.Convertible

import           Debug.Trace

import           Numeric.LinearAlgebra.Data   (toLists, cmap, flatten)
import           Numeric.LinearAlgebra        (maxIndex)
import qualified Numeric.LinearAlgebra.Static as SA
import qualified Numeric.LinearAlgebra.Devel  as U

import           Graphics.Gloss
import qualified Graphics.Gloss.Interface.IO.Interact as GI

import           Options.Applicative

import           Unsafe.Coerce 

import           Grenade
import           Grenade.Utils.OneHot
import           Grenade.Layers.Internal.Shrink (shrink_2d_rgba)

import           Graphics.Blank
import           Data.Text

type Coord = (Double, Double)

type Line = [Coord]

data MouseState = MouseUp | MouseDown deriving Eq

data CanvasState = CanvasState MouseState

loop :: DeviceContext -> CanvasState -> MNIST -> IO ()
loop context state net = do
  state' <- handleEvent context state net
  loop context state' net

handleEvent :: DeviceContext -> CanvasState -> MNIST -> IO CanvasState
handleEvent context state@(CanvasState mouseState) net = do 
  event <- wait context
  send context $ case (eType event, ePageXY event, mouseState) of
    ("mousedown", Just (x, y), _) -> do
      save ()
      moveTo (x, y)
      restore ()
      return $ CanvasState MouseDown
    
    ("mousemove", Just (x,y), MouseDown) -> do
      save ()
      lineTo (x, y)
      stroke ()
      restore ()
      return $ CanvasState mouseState

    ("mouseup", Just (x,y), _) -> do 
      imgdata <- getImageData (0, 0, 280, 280) 
      let ImageData s1 s2 v = imgdata
      traceM $ runNet'' net v

      return $ CanvasState MouseUp

    _ -> return $ state

runNet'' :: MNIST -> UB.Vector Word8 -> String
runNet'' net vec = runNet' net (DV.convert vec) 

runNet' :: MNIST -> V.Vector Word8 -> String
runNet' net m = (\(S1D ps) -> let (p, i) = (getProb . V.toList) (SA.extract ps)
                              in "This number is " ++ show i ++ " with probability " ++ show (p * 100) ++ "%") $ runNet net (conv m)
  where
    getProb :: (Show a, Ord a) => [a] -> (a, Int)
    getProb xs = maximumBy (comparing fst) (Prelude.zip xs [0..])

    conv :: V.Vector Word8 -> S ('D2 28 28)
    conv m = S2D $ fromMaybe (error "") $ SA.create $ shrink_2d_rgba 280 280 28 28 m 

type MNIST
  = Network
    '[ Convolution 1 10 5 5 1 1
     , Pooling 2 2 2 2
     , Relu
     , Convolution 10 16 5 5 1 1
     , Pooling 2 2 2 2
     , Reshape
     , Relu
     , FullyConnected 256 80
     , Logit
     , FullyConnected 80 10
     , Logit]
    '[ 'D2 28 28
     , 'D3 24 24 10
     , 'D3 12 12 10
     , 'D3 12 12 10
     , 'D3 8 8 16
     , 'D3 4 4 16
     , 'D1 256
     , 'D1 256
     , 'D1 80
     , 'D1 80
     , 'D1 10
     , 'D1 10]

data MNistLoadOpts = MNistLoadOpts FilePath -- Model path

mnist' :: Parser MNistLoadOpts
mnist' = MNistLoadOpts <$> argument str  (metavar "MODEL")

netLoad :: FilePath -> IO MNIST
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet (get :: Get MNIST) modelData

showShape' :: S ('D2 a b) -> String
showShape' (S2D mm) = 
  let m  = SA.extract mm
      ms = toLists m
      render n' | n' <= 0.2 * 255  = ' '
                | n' <= 0.4 * 255  = '.'
                | n' <= 0.6 * 255  = '-'
                | n' <= 0.8 * 255  = '='
                | otherwise =  '#'
      px = (fmap . fmap) render ms
  in Prelude.unlines px

draw :: S ('D2 224 224) -> Int -> Int -> S ('D2 224 224)
draw (S2D arr) x y = S2D $ fromMaybe (error "") $ SA.create m
  where 
    m = U.mapMatrixWithIndex f (SA.extract arr)

    f (x', y') p = if (x - x') ^ 2 + (y - y') ^ 2 <= 50 then 0 else p

main :: IO ()
main = do 
    MNistLoadOpts modelPath <- execParser (info (mnist' <**> helper) idm)
    
    net <- netLoad modelPath
    putStrLn "Successfully loaded model"
    
    putStrLn "Running... Go to localhost:3000"
    blankCanvas 3000 
                { events = ["mousedown", "mouseup", "mousemove"] } 
                (\context -> do
                               send context $ do lineWidth 30
                                                 strokeStyle "black"
                                                 lineCap RoundCap
                                                 lineJoin RoundCorner
                               loop context (CanvasState MouseUp) net)