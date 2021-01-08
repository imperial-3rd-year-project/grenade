{-# LANGUAGE CPP              #-}
{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-# LANGUAGE TemplateHaskell  #-}
{-# LANGUAGE TypeOperators    #-}

module Test.Grenade.Sys.Utils where

import           Numeric.LinearAlgebra        hiding (R, konst, randomVector,
                                               uniformSample, (===))
import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Data   as D
import           Numeric.LinearAlgebra.Static (L, R)
import qualified Numeric.LinearAlgebra.Static as H

import           Hedgehog
import qualified Hedgehog.Gen                 as Gen
import qualified Hedgehog.Range               as Range

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

import           GHC.TypeLits
import           Data.List
import           Data.Maybe
import           Data.Function
import           Data.Either
import           Control.Monad                (guard)
import           System.FilePath
import qualified Data.ByteString.Lazy as BS
import           Data.Binary

import           Grenade
import           Grenade.Utils.PascalVoc
import           Grenade.Utils.ImageNet

imagesDir :: FilePath
imagesDir = (takeDirectory __FILE__) </> "Images"

loadSerializedImage :: (KnownNat d, KnownNat c, KnownNat (d * c), c ~ 3) => FilePath -> IO (Maybe (S ('D3 d d c)))
loadSerializedImage path = do
  bs <- BS.readFile path
  let mat = decode bs :: LA.Matrix Double
  return $ S3D <$> (H.create (D.cmap doubleToRealNum mat))

loadSerializedChannel :: (KnownNat d, KnownNat c, KnownNat (d * c), c ~ 1) => FilePath -> IO (Maybe (S ('D3 d d c)))
loadSerializedChannel path = do
  bs <- BS.readFile path
  let mat = decode bs :: LA.Matrix Double
  return $ S3D <$> (H.create (D.cmap doubleToRealNum mat))

-- Performs linear regression on data points, where 
-- the x-values consist of the data set [1..n]
linearRegression :: Int -> [Double] -> Double
linearRegression n ys = m
  where
    n'    = fromIntegral n :: Double
    xs    = [1..n] :: [Int]
    x     = (n' + 1) / 2
    x2    = (fromIntegral $ sum (map (^ 2) xs)) / n'
    xyNum = (sum $ zipWith (\l i -> l * (fromIntegral i)) ys xs) :: Double
    xy    = xyNum / n' :: Double
    y     = (sum ys) / n'
    m     = (xy - (x * y)) / (x2 - (x * x)) :: Double
