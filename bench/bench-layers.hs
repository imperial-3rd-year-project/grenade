{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}

import           Criterion.Main

import           Control.Monad
import           GHC.TypeLits
import           System.Random

import           Grenade
import           Grenade.Utils.ListStore
import           Grenade.Utils.LinearAlgebra

import Data.Proxy
import Data.List (zipWith5)

import           Numeric.LinearAlgebra.Static      (R)
import qualified Numeric.LinearAlgebra.Static      as H

randomList :: (Double, Double) -> Int -> IO [Double]
randomList (a, b) n = replicateM n $ randomRIO (a, b)

randomBatchNormLayer :: forall c h w. (KnownNat c, KnownNat h, KnownNat w) 
                      => Bool -> IO (BatchNorm c h w 90)
randomBatchNormLayer training = do
  let rows     = fromIntegral $ natVal (Proxy :: Proxy h)
      columns  = fromIntegral $ natVal (Proxy :: Proxy w)
      channels = fromIntegral $ natVal (Proxy :: Proxy c)

  let (a, b) = (-2, 2)

  gamma'        <- randomList (a, b) channels
  beta'         <- randomList (a, b) channels
  running_mean' <- randomList (a, b) channels
  running_var'  <- randomList (a, b) channels

  let gamma        = H.fromList gamma'        :: R c
      beta         = H.fromList beta'         :: R c
      running_mean = H.fromList running_mean' :: R c
      running_var  = H.fromList running_var'  :: R c
      ε            = 0.00001

  return $ BatchNorm training (BatchNormParams gamma beta) running_mean running_var ε mkListStore

main :: IO ()
main = do
  let batchBenchSize1 = 32
      batchBenchSize2 = 128

  bnBig   :: BatchNorm 64 64 64 90 <- randomBatchNormLayer False
  bnSmall :: BatchNorm 10 3  4  90 <- randomBatchNormLayer False

  x :: S ('D3 64 64 64) <- randomOfShape
  y :: S ('D3 3 4 10)   <- randomOfShape

  let bigRun      = snd . runForwards bnBig   :: S ('D3 64 64 64) -> S ('D3 64 64 64)
  let smallRun    = snd . runForwards bnSmall :: S ('D3 3 4 10)   -> S ('D3 3 4 10)
  let oldBigRun   = run3DBatchNorm bnBig      :: S ('D3 64 64 64) -> S ('D3 64 64 64)
  let oldSmallRun = run3DBatchNorm bnSmall    :: S ('D3 3 4 10)   -> S ('D3 3 4 10)

  defaultMain
    [ bgroup
        "batchnorm forward 1D (CxHxW)"
        [ bench "batchnorm test forward 64x64x64"     $ nf bigRun      x
        , bench "batchnorm old test forward 64x64x64" $ nf oldBigRun   x
        , bench "batchnorm test forward 10x3x4"       $ nf smallRun    y
        , bench "batchnorm old test forward 10x3x4"   $ nf oldSmallRun y
        ]
    ]

-- REFERENCE 
run3DBatchNorm :: forall h w m c.
                  (KnownNat h, KnownNat w, KnownNat m, KnownNat c)
               => BatchNorm c h w m -> S ('D3 h w c) ->  S ('D3 h w c)
run3DBatchNorm (BatchNorm False (BatchNormParams gamma beta) runningMean runningVar ε _) inp
  = let ms     = vectorToList runningMean
        vs     = vectorToList runningVar
        gs     = vectorToList gamma
        bs     = vectorToList beta

        cs     = splitChannels inp :: [S ('D2 h w)]

        f c g b m v = let gs' = listToVector [g] :: R 1
                          bs' = listToVector [b] :: R 1
                          ms' = listToVector [m] :: R 1
                          vs' = listToVector [v] :: R 1
                          bn' = BatchNorm False (BatchNormParams gs' bs') ms' vs' ε undefined :: BatchNorm 1 h w m
                      in  runForwards bn' c

        (_, outs) = unzip $ zipWith5 f cs gs bs ms vs
      in combineChannels outs
