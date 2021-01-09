{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}

import           Data.Bits
import           Data.Convertible
import           Data.List                            (maximumBy)
import           Data.Maybe                           (fromJust)
import           Data.Ord
import qualified Data.Vector.Storable                 as V
import           Data.Word
import           Foreign.ForeignPtr

import           Numeric.LinearAlgebra
import qualified Numeric.LinearAlgebra.Devel          as U
import qualified Numeric.LinearAlgebra.Static         as SA

import           Graphics.Gloss
import           Graphics.Gloss.Interface.IO.Game
import qualified Graphics.Gloss.Interface.IO.Interact as GI

import           Grenade
import           Grenade.Demos.MNIST                  hiding (runNet')
import           Grenade.Layers.Internal.Shrink

data MouseState = MouseDown | MouseUp

-- | all of the information needed to create the canvas to draw on
data World      = Canvas (S ('D2 224 224)) MouseState MNIST

-- | turns the matrix in the world structure into a picture. To do this, we use some pointer 
--   magic to turn our matrix (which is represented internally as an array in memory) into 
--   a bitmap, where each pixel is represented by four word8s for the R, G, B, A channels
--   (which we construct as a Word32) 
renderCanvas :: World -> IO Picture
renderCanvas (Canvas (S2D !arr) _ _)
    = return $ bitmapOfForeignPtr 224 224 (BitmapFormat BottomToTop PxABGR) bitmapPtr False
  where
    -- | given a RealNum p, creates an RGBA word32 of the form (p, p, p, 255) - so it is essentially
    --   creating a greyscale pixel with no transparency.
    convColor :: RealNum -> Word32
    convColor p = let p'  =  convert p :: Word32
                      !w  =  unsafeShiftL p' 24
                         .|. unsafeShiftL p' 16
                         .|. unsafeShiftL p' 8
                         .|. 255
                  in w

    vec         = flatten $ SA.extract arr
    pixels      = V.map convColor vec
    (ptr, _, _) = U.unsafeToForeignPtr pixels
    -- | this last step is to turn the array of word32s into an array of word8s by
    --   interpreting each element as 4 word8s
    bitmapPtr   = castForeignPtr ptr

-- | updates the world with some event
handleInput :: GI.Event -> World -> IO World
-- if the mouse button is pressed down, update the mouse state 
handleInput (GI.EventKey (GI.MouseButton GI.LeftButton) (GI.Down) _ _) (Canvas arr _ net) 
  = return $ Canvas arr MouseDown net

-- if the mouse button is unpressed, update the mouse state and run a neural network inference
-- on the current canvas image by shrinking the image to the correct input size. Prints 
-- the result to the console.
handleInput (GI.EventKey (GI.MouseButton GI.LeftButton) (GI.Up) _ _) (Canvas (S2D arr') _ net) = 
  let extractedMatrix = SA.extract arr'
      shrunkenMatrix  = shrink_2d 224 224 28 28 extractedMatrix
      staticMatrix    = fromJust . fromStorableMatrix $ shrunkenMatrix
  in do 
    putStrLn $ runNet' net staticMatrix
    return $ Canvas (S2D arr') MouseUp net

-- If the 'c' key is pressed and then let go of, wipe the canvas
handleInput (GI.EventKey (GI.Char 'c') (GI.Up) _ _) (Canvas (S2D !arr) mb net) 
  = return $ Canvas cleanCanvas mb net
  where
    cleanCanvas = fromJust . fromStorableMatrix $ cleanMatrix
    cleanMatrix = U.mapMatrixWithIndex (const (const 255)) (SA.extract arr)

-- if moving the mouse while the left mouse button is held down, draw to the 
-- canvas (note that we add 112, since the origin of a canvas is at the middle)
handleInput (GI.EventMotion (y, x)) (Canvas arr MouseDown net) 
  = return $ Canvas (draw arr (xAsInt + 112) (yAsInt + 112)) MouseDown net
  where
    xAsInt = convert x
    yAsInt = convert y

handleInput _ c = return c

-- | draws a circle with squared radius 50 at the given coordinates
draw :: S ('D2 224 224) -> Int -> Int -> S ('D2 224 224)
draw (S2D arr) x y = fromJust . fromStorableMatrix $ m
  where
    m    = U.mapMatrixWithIndex f arr'
    arr' = SA.extract arr :: Matrix RealNum

    f :: (Int, Int) -> RealNum -> RealNum
    f (x', y') p = if (x - x') ^ (2 :: Int) + (y - y') ^ (2 :: Int) <= 50 then 0 else p

-- | runs a neural network inference and pretty prints the most probable label
--   with its probability
runNet' :: MNIST -> S ('D2 28 28) -> String
runNet' net m 
  = let S1D ps = runNet net m
        (p, i) = (getProb . V.toList) (SA.extract ps)
    in  "This number is " ++ show i ++ " with probability " ++ show (p * 100) ++ "%"
  where
    getProb :: [RealNum] -> (RealNum, Int)
    getProb xs = maximumBy (comparing fst) (zip xs [0..])

main :: IO ()
main = do
    mnistPath <- getPathForNetwork MNIST
    net <- (loadSerializedNetwork mnistPath :: IO MNIST)
    putStrLn "Successfully loaded model"

    let initialCanvas = Canvas initialMat MouseUp net

    playIO window backgroundColor 30 initialCanvas renderCanvas handleInput emptyStep
  where
    window            = InWindow "Draw here!" (224, 224) (100, 100)
    backgroundColor   = makeColor 255 255 255 0
    initialMat        = fromJust . fromStorable . V.fromList $ replicate (224 * 224) 255
    emptyStep _ world = return world
