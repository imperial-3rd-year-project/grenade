{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Networks.TinyYoloV2 where

import           Grenade.Core
import           Grenade.Layers

import           Grenade.Onnx

type TinyYoloV2 = Network
  '[ Convolution 'WithBias 'SameUpper 3 16 3 3 1 1
   , LeakyRelu
   , Lift (LoadActivation (Pooling 2 2 2 2))
   
   , Convolution 'WithBias 'SameUpper 16 32 3 3 1 1
   , LeakyRelu
   , Lift (LoadActivation (Pooling 2 2 2 2))

   , Convolution 'WithBias 'SameUpper 32 64 3 3 1 1
   , LeakyRelu
   , Lift (LoadActivation (Pooling 2 2 2 2))

   , Convolution 'WithBias 'SameUpper 64 128 3 3 1 1
   , LeakyRelu
   , Lift (LoadActivation (Pooling 2 2 2 2))

   , Convolution 'WithBias 'SameUpper 128 256 3 3 1 1
   , LeakyRelu
   , Lift (LoadActivation (Pooling 2 2 2 2))

   , Convolution 'WithBias 'SameUpper 256 512 3 3 1 1
   , LeakyRelu
   , PaddedPooling ('D3 13 13 512) ('D3 13 13 512) 2 2 1 1 0 0 1 1

   , Convolution 'WithBias 'SameUpper 512 1024 3 3 1 1
   , LeakyRelu

   , Convolution 'WithBias 'SameUpper 1024 1024 3 3 1 1
   , LeakyRelu

   , Convolution 'WithBias 'SameUpper 1024 125 1 1 1 1
   ]
  '[ 'D3 416 416 3                   -- Input
   , 'D3 416 416 16, 'D3 416 416 16  -- PaddedConv1, LeakyRelu
   , 'D3 208 208 16                  -- Pooling1

   , 'D3 208 208 32, 'D3 208 208 32  -- PaddedConv2, LeakyRelu
   , 'D3 104 104 32                  -- Pooling2

   , 'D3 104 104 64, 'D3 104 104 64  -- PaddedConv3, LeakyRelu
   , 'D3 52 52 64                    -- Pooling3

   , 'D3 52 52 128, 'D3 52 52 128    -- PaddedConv4, LeakyRelu
   , 'D3 26 26 128                   -- Pooling4

   , 'D3 26 26 256, 'D3 26 26 256    -- PaddedConv5, LeakyRelu
   , 'D3 13 13 256                   -- Pooling5

   , 'D3 13 13 512, 'D3 13 13 512    -- PaddedConv6, LeakyRelu
   , 'D3 13 13 512                   -- PaddedPooling6

   , 'D3 13 13 1024, 'D3 13 13 1024  -- PaddedConv7, LeakyRelu

   , 'D3 13 13 1024, 'D3 13 13 1024  -- PaddedConv8, LeakyRelu

   , 'D3 13 13 125                   -- BiasConv
   ]

loadTinyYoloV2 :: FilePath -> IO (Either OnnxLoadFailure TinyYoloV2)
loadTinyYoloV2 = loadOnnxModel
