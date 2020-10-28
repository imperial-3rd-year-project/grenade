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
import           Grenade.Canvas.Helpers

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

data MNistLoadOpts = MNistLoadOpts FilePath -- Model path

mnist' :: Parser MNistLoadOpts
mnist' = MNistLoadOpts <$> argument str  (metavar "MODEL")

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
