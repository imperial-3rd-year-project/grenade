{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
import           Control.Monad
import           Control.Monad.Random
import           Criterion.Main
import           Data.List                           (foldl')
import           Data.Singletons
import           Data.Singletons.Prelude.List
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.Static        as SA

import           Grenade
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra

ff :: SpecNet
ff = specFullyConnected 2 40 |=> specTanh1D 40 |=> netSpecInner |=> specFullyConnected 20 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specRelu1D 20 |=> specFullyConnected 20 10 |=> specRelu1D 10 |=> specFullyConnected 10 1 |=> specLogit1D 1 |=> specNil1D 1
  where netSpecInner = specFullyConnected 40 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specNil1D 20

hugeFf :: SpecNet
hugeFf = specFullyConnected 2 150 |=> specTanh1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specRelu1D 20 |=> specFullyConnected 20 10 |=> specRelu1D 10 |=> specFullyConnected 10 1 |=> specLogit1D 1 |=> specNil1D 1


main :: IO ()
main = do
  putStrLn $ "Benchmarking with type: " ++ nameF
  SpecConcreteNetwork1D1D netFF <- networkFromSpecificationWith Xavier ff
  SpecConcreteNetwork1D1D netHuge <- networkFromSpecificationWith Xavier hugeFf
  defaultMain
    [ bgroup
        "feedforward SGD"
        [ bench "ANN 1000 training steps" $ nfIO $ netTrain netFF defSGD 1000
        , bench "ANN 10000 training steps" $ nfIO $ netTrain netFF defSGD 10000
        , bench "ANN Huge 100 train steps" $ nfIO $ netTrain netHuge defSGD 100
        , bench "ANN Huge 1000 train steps" $ nfIO $ netTrain netHuge defSGD 1000
        ]
    , bgroup
        "feedforward Adam"
        [ bench "ANN 1000 training steps" $ nfIO $ netTrain netFF defAdam 1000
        , bench "ANN 10000 training steps" $ nfIO $ netTrain netFF defAdam 10000
        , bench "ANN Huge 100 train steps" $ nfIO $ netTrain netHuge defAdam 100
        , bench "ANN Huge 1000 train steps" $ nfIO $ netTrain netHuge defAdam 1000
        ]
    ]
  putStrLn $ "Benchmarked with type: " ++ nameF

testRun2D :: Pad 1 1 1 1 -> S ('D2 60 60) -> S ('D2 62 62)
testRun2D = snd ... runForwards

testRun3D :: Pad 1 1 1 1 -> S ('D3 60 60 1) -> S ('D3 62 62 1)
testRun3D = snd ... runForwards

testRun2D' :: Crop 1 1 1 1 -> S ('D2 60 60) -> S ('D2 58 58)
testRun2D' = snd ... runForwards

testRun3D' :: Crop 1 1 1 1 -> S ('D3 60 60 1) -> S ('D3 58 58 1)
testRun3D' = snd ... runForwards

(...) :: (a -> b) -> (c -> d -> a) -> c -> d -> b
(...) = (.) . (.)

netTrain ::
     (SingI (Last shapes), MonadRandom m, KnownNat len1, KnownNat len2, Head shapes ~ 'D1 len1, Last shapes ~ 'D1 len2)
  => Network layers shapes
  -> Optimizer o
  -> Int
  -> m (Network layers shapes)
netTrain net0 op n = do
  inps <-
    replicateM n $ do
      s <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
  let outs =
        flip map inps $ \(S1D v) ->
          if v `inCircle` (fromRational 0.50, 0.50) || v `inCircle` (fromRational (-0.50), 0.50)
            then S1D $ fromRational 1
            else S1D $ fromRational 0
  let trained = foldl' trainEach net0 (zip inps outs)
  return trained
  where
    trainEach !network (i, o) = fst $ train op network i o quadratic'
    inCircle :: KnownNat n => SA.R n -> (SA.R n, RealNum) -> Bool
    v `inCircle` (o, r) = SA.norm_2 (v - o) <= r
