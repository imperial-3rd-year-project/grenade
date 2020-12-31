{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NoStarIsType          #-}
{-# LANGUAGE OverloadedLabels      #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}

import           Control.Monad
import           System.Random.MWC

import           Data.Constraint              (Dict (..))
import           Data.List                    (zipWith5)
import           Data.Proxy
import           Data.Reflection
import           Data.Singletons
import           Data.Singletons.TypeLits     hiding (natVal)
import           GHC.TypeLits
import           Unsafe.Coerce                (unsafeCoerce)

import           Numeric.LinearAlgebra.Static (L, R)
import qualified Numeric.LinearAlgebra.Static as H

import           Criterion.Main

import           Grenade
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

main :: IO ()
main = do
  defaultMain
    [ bgroup
        "batchnorm forward (CxHxW)"
        [ benchBatchNorm "1x1x64" 1 1 64
        , benchBatchNorm "1x64x64" 1 64 64
        , benchBatchNorm "64x64x64" 64 64 64
        ]
    , bgroup
        "convolutions with bias (CxHxW)"
        [ benchBiasConvolution "1024x64x64, 125 kernels" 1024 64 64 125 1 1 64 64
        , benchBiasConvolution "3x416x416, 64 kernels" 3 416 416 64 2 1 208 208
        ]
    , bgroup
        "leaky relu (CxHxW)"
        [ benchLeakyRelu "3x416x416" 3 416 416
        , benchLeakyRelu "1024x16x16" 1024 16 16
        ]
    ]

-- BENCHMARK GENERATION FUNCTIONS

benchBiasConvolution :: String -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Benchmark
benchBiasConvolution name channels width height filters strides kernels outWidth outHeight
  | strides * (outHeight - 1) <= (height - kernels + 1) - 1 && (height - kernels + 1) <= (outHeight * strides) && strides * (outWidth - 1) <= (width - kernels + 1) - 1 && (width - kernels + 1) <= (outWidth * strides)
      = case (someNatVal (fromIntegral channels), someNatVal (fromIntegral width), someNatVal (fromIntegral height), someNatVal (fromIntegral filters), someNatVal (fromIntegral strides), someNatVal (fromIntegral kernels), someNatVal (fromIntegral outWidth), someNatVal (fromIntegral outHeight)) of
          (Just (SomeNat (Proxy :: Proxy channels)), Just (SomeNat (Proxy :: Proxy width)), Just (SomeNat (Proxy :: Proxy height)), Just (SomeNat (Proxy :: Proxy filters)), Just (SomeNat (Proxy :: Proxy strides)), Just (SomeNat (Proxy :: Proxy kernels)), Just (SomeNat (Proxy :: Proxy outWidth)), Just (SomeNat (Proxy :: Proxy outHeight))) ->
            case (channels, width, height
                 , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strides * (outHeight - 1) <= (height - kernels + 1) - 1 ) )
                 , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (height - kernels + 1) <= (outHeight * strides) ) )
                 , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strides * (outWidth - 1) <= (width - kernels + 1) - 1 ) )
                 , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (width - kernels + 1) <= (outWidth * strides) ) ) ) of
              (1, 1, _, _, _, _, _)
                -> error "1D convolutions are not allowed"
              (1, _, _, Dict, Dict, Dict, Dict)
                -> env (generateBiasConvEnv :: IO (Convolution 'WithBias 1 filters kernels kernels strides strides, S ('D2 height width))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D2 height width) -> S ('D3 outHeight outWidth filters )) x
              (_, _, _, Dict, Dict, Dict, Dict)
                -> env (generateBiasConvEnv :: IO (Convolution 'WithBias channels filters kernels kernels strides strides, S ('D3 height width channels))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D3 height width channels) -> S ('D3 outHeight outWidth filters)) x
  where
    generateBiasConvEnv :: forall channels filters kernel1 kernel2 strides1 strides2 s.
                            ( SingI s, KnownNat channels, KnownNat filters, KnownNat kernel1, KnownNat kernel2,
                              KnownNat strides1, KnownNat strides2 )
                            => IO (Convolution 'WithBias channels filters kernel1 kernel2 strides1 strides2, S s)
    generateBiasConvEnv = do
      x     <- randomOfShape
      gen   <- createSystemRandom
      layer <- createRandomWith UniformInit gen
      return (layer, x)

benchBatchNorm :: String -> Int -> Int -> Int -> Benchmark
benchBatchNorm name channels width height
  = case (someNatVal (fromIntegral channels), someNatVal (fromIntegral width), someNatVal (fromIntegral height)) of
          (Just (SomeNat (Proxy :: Proxy channels)), Just (SomeNat (Proxy :: Proxy width)), Just (SomeNat (Proxy :: Proxy height))) ->
            case (channels, width, height) of
              (1, 1, _) -> env (generateBatchNormEnv False :: IO (BatchNorm 1 1 width 90, S ('D1 width))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D1 width) -> S ('D1 width)) x
              (1, _, _) -> env (generateBatchNormEnv False :: IO (BatchNorm 1 width height 90, S ('D2 width height))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D2 width height) -> S ('D2 width height)) x
              (_, _, _) -> env (generateBatchNormEnv False :: IO (BatchNorm channels width height 90, S ('D3 width height channels))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D3 width height channels) -> S ('D3 width height channels)) x
  where
    generateBatchNormEnv :: forall c h w s. (KnownNat c, KnownNat h, KnownNat w, SingI s)
                         => Bool -> IO (BatchNorm c h w 90, S s)
    generateBatchNormEnv training = do
      x     <- randomOfShape
      gens  <- replicateM 4 createSystemRandom
      seeds <- mapM uniform gens :: IO [Int]
      let [gamma, beta, running_mean, running_var] = map (\s -> H.randomVector s H.Uniform) seeds :: [R c]
          ε            = 0.00001
      return (BatchNorm training (BatchNormParams gamma beta) running_mean running_var ε mkListStore, x)

benchLeakyRelu :: String -> Int -> Int -> Int -> Benchmark
benchLeakyRelu name channels width height
  = case (someNatVal (fromIntegral channels), someNatVal (fromIntegral width), someNatVal (fromIntegral height)) of
          (Just (SomeNat (Proxy :: Proxy channels)), Just (SomeNat (Proxy :: Proxy width)), Just (SomeNat (Proxy :: Proxy height))) ->
            case (channels, width, height) of
              (1, 1, _) -> env (generateLeakyReluEnv :: IO (LeakyRelu, S ('D1 width))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D1 width) -> S ('D1 width)) x
              (1, _, _) -> env (generateLeakyReluEnv :: IO (LeakyRelu, S ('D2 width height))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D2 width height) -> S ('D2 width height)) x
              (_, _, _) -> env (generateLeakyReluEnv :: IO (LeakyRelu, S ('D3 width height channels))) $ \ ~(layer, x) -> bench name $ nf (snd . runForwards layer :: S ('D3 width height channels) -> S ('D3 width height channels)) x
  where
    generateLeakyReluEnv :: forall s. (SingI s) => IO (LeakyRelu, S s)
    generateLeakyReluEnv = do
      x     <- randomOfShape
      gen   <- createSystemRandom
      alpha <- uniformR (-1, 1) gen :: IO RealNum
      return (LeakyRelu alpha, x)