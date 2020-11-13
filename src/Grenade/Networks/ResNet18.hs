{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Networks.ResNet18 where

import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Layers


type ResNet18BranchRight channels size
  = Network [ Convolution channels channels 3 3 1 1, Pad 1 1 1 1, BatchNorm channels size size 90, Relu, Convolution channels channels 3 3 1 1, Pad 1 1 1 1, BatchNorm channels size size 90 ]
            [ 'D3 size size channels, 'D3 (size - 2) (size - 2) channels, 'D3 size size channels, 'D3 size size channels, 'D3 size size channels, 'D3 (size - 2) (size - 2) channels, 'D3 size size channels, 'D3 size size channels ]

type ResNet18BranchShrinkRight inChannels outChannels inSize outSize
  = Network [ Convolution inChannels outChannels 3 3 2 2, Pad 1 1 1 1, BatchNorm outChannels outSize outSize 90, Relu, Convolution outChannels outChannels 3 3 1 1, Pad 1 1 1 1, BatchNorm outChannels outSize outSize 90 ]
            [ 'D3 inSize inSize inChannels, 'D3 (outSize - 2) (outSize - 2) outChannels, 'D3 outSize outSize outChannels, 'D3 outSize outSize outChannels, 'D3 outSize outSize outChannels, 'D3 (outSize - 2) (outSize - 2) outChannels, 'D3 outSize outSize outChannels, 'D3 outSize outSize outChannels ]

type ResNet18BranchShrinkLeft inChannels outChannels inSize outSize
  = Network [ Convolution inChannels outChannels 1 1 2 2, BatchNorm outChannels outSize outSize 90 ]
            [ 'D3 inSize inSize inChannels, 'D3 outSize outSize outChannels, 'D3 outSize outSize outChannels ]

type ResNet18Block size channels
  = Network [Merge Trivial (ResNet18BranchRight size channels), Relu]
            ['D3 size size channels, 'D3 size size channels, 'D3 size size channels ]

type ResNet18ShrinkBlock inSize outSize inChannels outChannels
  = Network [Merge (ResNet18BranchShrinkLeft inSize outSize inChannels outChannels) (ResNet18BranchShrinkRight inSize outSize inChannels outChannels), Relu]
            ['D3 inSize inSize inChannels, 'D3 outSize outSize outChannels, 'D3 outSize outSize outChannels ]


type ResNet18 
  = Network 
      [ Convolution 3 64 7 7 2 2
      , Pad 3 3 3 3
      , Relu
      , Pooling 3 3 2 2
      , ResNet18Block 56 64
      , ResNet18Block 56 64
      , ResNet18ShrinkBlock 56 28 64 128
      , ResNet18Block 28 128
      , ResNet18ShrinkBlock 28 14 128 256
      , ResNet18Block 14 256
      , ResNet18ShrinkBlock 14 7 256 512
      , ResNet18Block 7 512
      , GlobalAvgPool
      , Reshape
      , FullyConnected 512 1000
      ]
      [ 'D3 224 224 3
      , 'D3 106 106 64 
      , 'D3 112 112 64 
      , 'D3 112 112 64 
      , 'D3 56  56  64  -- first block in
      , 'D3 56  56  64  -- 2 in
      , 'D3 56  56  64  -- 3 in
      , 'D3 28  28  128 -- 4 in
      , 'D3 28  28  128 -- 5 in
      , 'D3 14  14  256 -- 6 in
      , 'D3 14  14  256 -- 7 in
      , 'D3 7   7   512 -- 8 in
      , 'D3 7   7   512
      , 'D3 1   1   512
      , 'D1 512
      , 'D1 1000
      ]