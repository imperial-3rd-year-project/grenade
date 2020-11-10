{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}

import           Criterion.Main

import           Control.Monad
import           Data.Proxy
import           GHC.TypeLits
import           System.Random

import           Grenade
import           Grenade.Layers.BatchNormalisation
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

import           Numeric.LinearAlgebra.Static      (L, R)
import qualified Numeric.LinearAlgebra.Static      as H

randomList :: (Double, Double) -> Int -> IO [Double]
randomList (a, b) n = replicateM n $ randomRIO (a, b)

randomBatchNormLayer :: forall n. KnownNat n => Bool -> IO (BatchNorm 1 1 n 90)
randomBatchNormLayer training = do
  let l = fromIntegral $ natVal (Proxy :: Proxy n)

  gamma'        <- randomList (-2, 2) l
  beta'         <- randomList (-2, 2) l
  running_mean' <- randomList (-2, 2) l
  running_var'  <- randomList (-2, 2) l

  let gamma        = H.fromList gamma'        :: R n
      beta         = H.fromList beta'         :: R n
      running_mean = H.fromList running_mean' :: R n
      running_var  = H.fromList running_var'  :: R n

  return $ BatchNorm training (BatchNormParams gamma beta) running_mean running_var mkListStore

main :: IO ()
main = do
  let batchBenchSize1 = 32
      batchBenchSize2 = 128

  bn5   :: BatchNorm 1 1 5   90 <- randomBatchNormLayer True 
  bn250 :: BatchNorm 1 1 250 90 <- randomBatchNormLayer True 

  xs1D :: [S ('D1 5)]   <- sequence $ take batchBenchSize1 $ repeat randomOfShape
  ys1D :: [S ('D1 250)] <- sequence $ take batchBenchSize1 $ repeat randomOfShape

  xs'1D :: [S ('D1 5)]   <- sequence $ take batchBenchSize2 $ repeat randomOfShape
  ys'1D :: [S ('D1 250)] <- sequence $ take batchBenchSize2 $ repeat randomOfShape

  let f5 = runBatchForwards bn5     :: [S ('D1 5)] -> ([Tape (BatchNorm 1 1 5 90) ('D1 5) ('D1 5)], [S ('D1 5)])
  let f250 = runBatchForwards bn250 :: [S ('D1 250)] -> ([Tape (BatchNorm 1 1 250 90) ('D1 250) ('D1 250)], [S ('D1 250)])

  defaultMain
    [ bgroup
        "batchnorm forward 1D"
        [ bench "batchnorm forward 32 of 5"    $ nf f5   xs1D
        , bench "batchnorm forward 32 of 250"  $ nf f250 ys1D
        , bench "batchnorm forward 128 of 5"   $ nf f5   xs'1D
        , bench "batchnorm forward 128 of 250" $ nf f250 ys'1D
        ]
    ]

