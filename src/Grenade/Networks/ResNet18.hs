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

import           Grenade.Core
import           Grenade.Layers

import           Grenade.Onnx.ActivationLayer
import           Grenade.Onnx.BypassLayer
import           Grenade.Onnx.Iso
import           Grenade.Onnx.LoadNetwork()
import           Grenade.Onnx.OnnxLoadable
import           Grenade.Onnx.ParallelLayer

type ResNet18BranchRight size channels
  = Network
      '[ PaddedConvolution ('D3 size size channels) ('D3 size size channels) channels channels 3 3 1 1 1 1 1 1
       , BatchNorm channels size size 90
       , Lift (LoadActivation Relu)
       , PaddedConvolution ('D3 size size channels) ('D3 size size channels) channels channels 3 3 1 1 1 1 1 1
       , BatchNorm channels size size 90
       ]
      '[ 'D3 size size channels
       , 'D3 size size channels
       , 'D3 size size channels
       , 'D3 size size channels
       , 'D3 size size channels
       , 'D3 size size channels
       ]

type ResNet18BranchShrinkRight inSize outSize inChannels outChannels
  = Network
      '[ PaddedConvolution ('D3 inSize inSize inChannels) ('D3 outSize outSize outChannels) inChannels outChannels 3 3 2 2 1 1 1 1
       , BatchNorm outChannels outSize outSize 90
       , Lift (LoadActivation Relu)
       , PaddedConvolution ('D3 outSize outSize outChannels)('D3 outSize outSize outChannels) outChannels outChannels 3 3 1 1 1 1 1 1
       , BatchNorm outChannels outSize outSize 90
       ]
      '[ 'D3 inSize inSize inChannels
       , 'D3 outSize outSize outChannels
       , 'D3 outSize outSize outChannels
       , 'D3 outSize outSize outChannels
       , 'D3 outSize outSize outChannels
       , 'D3 outSize outSize outChannels
       ]

type ResNet18BranchShrinkLeft inSize outSize inChannels outChannels
  = Network
      '[ PaddedConvolution ('D3 inSize inSize inChannels) ('D3 outSize outSize outChannels) inChannels outChannels 1 1 2 2 0 0 0 0
       , BatchNorm outChannels outSize outSize 90
       ]
      '[ 'D3 inSize inSize inChannels
       , 'D3 outSize outSize outChannels
       , 'D3 outSize outSize outChannels
       ]

type ResNet18Block size channels
  = Network
      '[ Lift (LoadParallel (Merge (Lift (LoadBypass Trivial)) (ResNet18BranchRight size channels))), Lift (LoadActivation Relu)]
      '[ 'D3 size size channels, 'D3 size size channels, 'D3 size size channels ]

type ResNet18ShrinkBlock inSize outSize inChannels outChannels
  = Network
      '[ Lift (LoadParallel (Merge (ResNet18BranchShrinkLeft inSize outSize inChannels outChannels) (ResNet18BranchShrinkRight inSize outSize inChannels outChannels))), Lift (LoadActivation Relu)]
      '[ 'D3 inSize inSize inChannels, 'D3 outSize outSize outChannels, 'D3 outSize outSize outChannels ]

type ResNetTest
  = Network
      '[ PaddedConvolution ('D3 224 224 3) ('D3 112 112 64 ) 3 64 7 7 2 2 3 3 3 3
       , Lift (LoadActivation Relu)
       ]
      '[ 'D3 224 224 3
       , 'D3 112 112 64
       , 'D3 112 112 64
       ]

type ResNet18
  = Network
     '[ PaddedConvolution ('D3 224 224 3) ('D3 112 112 64 ) 3 64 7 7 2 2 3 3 3 3
      , BatchNorm 64 112 112 90
      , Lift (LoadActivation Relu)
      , PaddedPooling ('D3 112 112 64) ('D3 56 56 64) 3 3 2 2 1 1 1 1
      , ResNet18Block 56 64
      , ResNet18Block 56 64
      , ResNet18ShrinkBlock 56 28 64 128
      , ResNet18Block 28 128
      , ResNet18ShrinkBlock 28 14 128 256
      , ResNet18Block 14 256
      , ResNet18ShrinkBlock 14 7 256 512
      , ResNet18Block 7 512
      , Lift (LoadActivation GlobalAvgPool)
      , Lift (LoadActivation Reshape)
      , FullyConnected 512 1000
      ]
     '[ 'D3 224 224 3
      , 'D3 112 112 64
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

loadResNet :: FilePath -> IO (Maybe ResNet18)
loadResNet = loadOnnxModel
