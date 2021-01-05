{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Networks
Description : SuperResolution network type definition
Maintainer  : Theo Charalambous
License     : BSD2
Stability   : experimental

Definition of the SuperResolution v10 model compatible with the onnx file found at 
<https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx>
-}

module Grenade.Networks.SuperResolution 
  (
    SuperResolution
  , loadSuperResolution
  )
where

import           Grenade.Core
import           Grenade.Layers

import           Grenade.Onnx.ActivationLayer
import           Grenade.Onnx.Iso
import           Grenade.Onnx.Model
import           Grenade.Onnx.OnnxLoadFailure

type SuperResolution
  = Network
     '[ Convolution 'WithBias ('Padding 2 2 2 2) 1 64 5 5 1 1
      , Lift (LoadActivation Relu)
      , Convolution 'WithBias ('Padding 1 1 1 1) 64 64 3 3 1 1
      , Lift (LoadActivation Relu)
      , Convolution 'WithBias ('Padding 1 1 1 1) 64 32 3 3 1 1
      , Lift (LoadActivation Relu)
      , Convolution 'WithBias ('Padding 1 1 1 1) 32 9 3 3 1 1
      , Lift (LoadActivation Reshape) 
      , Transpose 4 ('D4 3 3 224 224) ('D4 224 3 224 3)
      , Lift (LoadActivation Reshape)
      ]
     '[ 'D3 224 224 1
      , 'D3 224 224 64
      , 'D3 224 224 64
      , 'D3 224 224 64
      , 'D3 224 224 64
      , 'D3 224 224 32
      , 'D3 224 224 32
      , 'D3 224 224 9
      , 'D4 3 3 224 224
      , 'D4 224 3 224 3
      , 'D3 672 672 1
      ]

loadSuperResolution :: FilePath -> IO (Either OnnxLoadFailure SuperResolution)
loadSuperResolution = loadOnnxModel
