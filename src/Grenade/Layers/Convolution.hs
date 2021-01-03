{-# LANGUAGE CPP                   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels      #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Grenade.Layers.Convolution where

import           Data.Function                       ((&))
import           Data.Kind                           (Type)
import           Data.List                           (foldl1')
import           Data.Maybe
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits
import           Control.DeepSeq
import           Lens.Micro                          ((^.))

import           Numeric.LinearAlgebra               hiding (R, konst,
                                                      uniformSample)
import qualified Numeric.LinearAlgebra               as LA
import           Numeric.LinearAlgebra.Static        hiding (build, toRows, (&),
                                                      (|||), size)

import           Grenade.Core
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Update
import           Grenade.Onnx
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

type OutputShapeIsOkay (input :: Nat) (pad :: Nat) (kernel :: Nat) (strides :: Nat) (output :: Nat)
  = ( strides * (output - 1) <= (input - kernel + pad)
    , (input - kernel + pad) <= (output * strides ) - 1 )

data HasBias = WithBias | WithoutBias

data ConvPadding = NoPadding | SameUpper | SameLower | Padding Nat Nat Nat Nat

data Convolution :: HasBias
                 -> ConvPadding
                 -> Nat -- Number of channels, for the first layer this could be RGB for instance.
                 -> Nat -- Number of filters, this is the number of channels output by the layer.
                 -> Nat -- The number of rows in the kernel filter
                 -> Nat -- The number of column in the kernel filter
                 -> Nat -- The row stride of the convolution filter
                 -> Nat -- The columns stride of the convolution filter
                 -> Type where
  Convolution :: ( KnownNat channels
                 , KnownNat filters
                 , KnownNat kernelRows
                 , KnownNat kernelColumns
                 , KnownNat strideRows
                 , KnownNat strideColumns
                 , KnownNat kernelFlattened
                 , kernelFlattened ~ (kernelRows * kernelColumns * channels))
              => !(L kernelFlattened filters) -- The kernel filter weights
              -> !(ListStore (L kernelFlattened filters)) -- The last kernel update (or momentum)
              -> Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns

  BiasConvolution :: ( KnownNat channels
                     , KnownNat filters
                     , KnownNat kernelRows
                     , KnownNat kernelColumns
                     , KnownNat strideRows
                     , KnownNat strideColumns
                     , KnownNat kernelFlattened
                     , kernelFlattened ~ (kernelRows * kernelColumns * channels))
                  => !(L kernelFlattened filters) -- The kernel filter weights
                  -> !(R filters) -- The bias weights
                  -> !(ListStore (L kernelFlattened filters)) -- The last kernel update (or momentum)
                  -> Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns

data Convolution' :: HasBias
                  -> ConvPadding
                  -> Nat -- Number of channels, for the first layer this could be RGB for instance.
                  -> Nat -- Number of filters, this is the number of channels output by the layer.
                  -> Nat -- The number of rows in the kernel filter
                  -> Nat -- The number of column in the kernel filter
                  -> Nat -- The row stride of the convolution filter
                  -> Nat -- The columns stride of the convolution filter
                  -> Type where
  Convolution' :: ( KnownNat channels
                  , KnownNat filters
                  , KnownNat kernelRows
                  , KnownNat kernelColumns
                  , KnownNat strideRows
                  , KnownNat strideColumns
                  , KnownNat kernelFlattened
                  , kernelFlattened ~ (kernelRows * kernelColumns * channels))
               => !(L kernelFlattened filters) -- The kernel filter weights
               -> Convolution' 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns

  BiasConvolution' :: ( KnownNat channels
                      , KnownNat filters
                      , KnownNat kernelRows
                      , KnownNat kernelColumns
                      , KnownNat strideRows
                      , KnownNat strideColumns
                      , KnownNat kernelFlattened
                      , kernelFlattened ~ (kernelRows * kernelColumns * channels))
                   => !(L kernelFlattened filters) -- The kernel filter weights
                   -> !(R filters) -- The bias weights
                   -> Convolution' 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns

{---------------------------}
{--    Layer instances    --}
{---------------------------}

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat ((kernelRows * kernelColumns) * channels)
         , KnownNat (filters * ((kernelRows * kernelColumns) * channels))
         ) => RandomLayer (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  createRandomWith m gen = do
    wN <- getRandomMatrix i i m gen
    wB <- getRandomVector i i m gen
    return $ BiasConvolution wN wB mkListStore
    where
      i = natVal (Proxy :: Proxy ((kernelRows * kernelColumns) * channels))

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat ((kernelRows * kernelColumns) * channels)
         , KnownNat (filters * ((kernelRows * kernelColumns) * channels))
         ) => RandomLayer (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  createRandomWith m gen = do
    wN <- getRandomMatrix i i m gen
    return $ Convolution wN mkListStore
    where
      i = natVal (Proxy :: Proxy ((kernelRows * kernelColumns) * channels))

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         , kernelFlattened ~ (kernelRows * kernelColumns * channels)
         ) => UpdateLayer (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) = (Convolution' 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns)

  type MomentumStore (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns)  = ListStore (L (kernelRows * kernelColumns * channels) filters)

  runUpdate opt@OptSGD{} x@(Convolution oldKernel store) (Convolution' kernelGradient) =
    let  momentum = getData opt x store
         result = descendMatrix opt (MatrixValuesSGD oldKernel kernelGradient momentum)
         newStore = setData opt x store (matrixMomentum result)
    in Convolution (matrixActivations result) newStore
  runUpdate opt@OptAdam{} x@(Convolution oldKernel store) (Convolution' kernelGradient) =
    let (m, v) = toTuple $ getData opt x store
        result = descendMatrix opt (MatrixValuesAdam (getStep store) oldKernel kernelGradient m v)
        newStore = setData opt x store [matrixM result, matrixV result]
    in Convolution (matrixActivations result) newStore
    where toTuple [m ,v] = (m, v)
          toTuple xs = error $ "unexpected input of length " ++ show (length xs) ++ "in toTuple in Convolution.hs"

  reduceGradient grads = Convolution' $ dmmap (/ (fromIntegral $ length grads)) (foldl1' add (map (\(Convolution' x) -> x) grads))

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         , kernelFlattened ~ (kernelRows * kernelColumns * channels)
         ) => UpdateLayer (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) = (Convolution' 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns)

  type MomentumStore (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns)  = ListStore (L (kernelRows * kernelColumns * channels) filters)

  runUpdate      = undefined
  reduceGradient = undefined


{-----------------------------------}
{--    No Bias Layer instances    --}
{-----------------------------------}

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
--   Case without bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithoutBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Convolution 'WithoutBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Convolution kernel _) (S3D input) =     
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)

        out  = forwardConv2d ex cs ix iy ek fs kx ky sx sy ox oy 0 0
    in  (S3D input, S3D . fromJust . create $ out)

  runBackwards (Convolution kernel _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)

        (dx, dw)  = backwardConv2d ex cs ix iy ek fs kx ky sx sy 0 0 0 0 ey ox oy
    in  (Convolution' . fromJust . create $ dw, S3D . fromJust . create $ dx)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , KnownNat padt
         , KnownNat padl
         , KnownNat padb
         , KnownNat padr
         , OutputShapeIsOkay inputRows (padt + padb) kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols (padl + padr) kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithoutBias ('Padding padl padt padr padb) channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Convolution 'WithoutBias ('Padding padl padt padr padb) channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Convolution kernel _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)
        pl = fromIntegral $ natVal (Proxy :: Proxy padl)
        pt = fromIntegral $ natVal (Proxy :: Proxy padt)

        out  = fromJust . create $ forwardConv2d ex cs ix iy ek fs kx ky sx sy ox oy pl pt
    in  (S3D input, S3D out)

  runBackwards (Convolution kernel _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )
        pl = fromIntegral $ natVal (Proxy :: Proxy padl      )
        pt = fromIntegral $ natVal (Proxy :: Proxy padt      )
        pr = fromIntegral $ natVal (Proxy :: Proxy padr      )
        pb = fromIntegral $ natVal (Proxy :: Proxy padb      )

        (dx, dw) = backwardConv2d ex cs ix iy ek fs kx ky sx sy pl pt pr pb ey ox oy
    in  (Convolution' . fromJust . create $ dw, S3D . fromJust . create $ dx)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithoutBias 'SameUpper channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) where

  type Tape (Convolution 'WithoutBias 'SameUpper channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Convolution kernel _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padt = div pady 2
        padl = div padx 2

        out  = fromJust . create $ forwardConv2d ex cs ix iy ek fs kx ky sx sy ix iy padl padt
    in  (S3D input, S3D out)

  runBackwards (Convolution kernel _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padl = div padx 2
        padt = div pady 2
        padr = padx - padl
        padb = pady - padt

        (dx, dw) = backwardConv2d ex cs ix iy ek fs kx ky sx sy padl padt padr padb ey ix iy
    in  (Convolution' . fromJust . create $ dw, S3D . fromJust . create $ dx)

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
--   Case without bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithoutBias 'SameLower channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) where

  type Tape (Convolution 'WithoutBias 'SameLower channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Convolution kernel _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padt = pady - div pady 2
        padl = padx - div padx 2
        out  = fromJust . create $ forwardConv2d ex cs ix iy ek fs kx ky sx sy ix iy padl padt

    in  (S3D input, S3D out)

  runBackwards (Convolution kernel _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padt = pady - padb
        padl = padx - padr
        padr = div padx 2
        padb = div pady 2

        (dx, dw) = backwardConv2d ex cs ix iy ek fs kx ky sx sy padl padt padr padb ey ix iy
    in  (Convolution' . fromJust . create $ dw, S3D . fromJust . create $ dx)

{--------------------------------}
{--    Bias Layer instances    --}
{--------------------------------}

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
--   Case with bias vector
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Convolution 'WithBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (BiasConvolution kernel biases _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        eb = extract biases
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)

        out  = fromJust . create $ forwardBiasConv2d ex cs ix iy eb ek fs kx ky sx sy ox oy 0 0

    in  (S3D input, S3D out)

  runBackwards (BiasConvolution kernel _ _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)
        
        (dx', dw', db') = backwardBiasConv2d ex cs ix iy ek fs kx ky sx sy 0 0 0 0 ey ox oy
        dx              = fromJust . create $ dx'
        dw              = fromJust . create $ dw'
        db              = fromJust . create $ db'
    in  (BiasConvolution' dw db, S3D dx)


instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , KnownNat padt
         , KnownNat padl
         , KnownNat padb
         , KnownNat padr
         , OutputShapeIsOkay inputRows (padt + padb) kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols (padl + padr) kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithBias ('Padding padl padt padr padb) channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Convolution 'WithBias ('Padding padl padt padr padb) channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (BiasConvolution kernel biases _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        eb = extract biases
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )
        pl = fromIntegral $ natVal (Proxy :: Proxy padl      )
        pt = fromIntegral $ natVal (Proxy :: Proxy padt      )

        out  = fromJust . create $ forwardBiasConv2d ex cs ix iy eb ek fs kx ky sx sy ox oy pl pt

    in  (S3D input, S3D out)

  runBackwards (BiasConvolution kernel _ _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )
        pl = fromIntegral $ natVal (Proxy :: Proxy padl      )
        pt = fromIntegral $ natVal (Proxy :: Proxy padt      )
        pr = fromIntegral $ natVal (Proxy :: Proxy padr      )
        pb = fromIntegral $ natVal (Proxy :: Proxy padb      )

        (dx', dw', db') = backwardBiasConv2d ex cs ix iy ek fs kx ky sx sy pl pt pr pb ey ox oy
        dx              = fromJust . create $ dx'
        dw              = fromJust . create $ dw'
        db              = fromJust . create $ db'
    in  (BiasConvolution' dw db, S3D dx)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithBias 'SameUpper channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) where

  type Tape (Convolution 'WithBias 'SameUpper channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (BiasConvolution kernel bias _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        eb = extract bias
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padt = div pady 2
        padl = div padx 2

        out  = fromJust . create $ forwardBiasConv2d ex cs ix iy eb ek fs kx ky sx sy ix iy padl padt

    in  (S3D input, S3D out)

  runBackwards (BiasConvolution kernel _ _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padl = div padx 2
        padt = div pady 2
        padr = padx - padl
        padb = pady - padt

        (dx', dw', db') = backwardBiasConv2d ex cs ix iy ek fs kx ky sx sy padl padt padr padb ey ix iy
        dx              = fromJust . create $ dx'
        dw              = fromJust . create $ dw'
        db              = fromJust . create $ db'
    in  (BiasConvolution' dw db, S3D dx)

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
--   Case without bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithBias 'SameLower channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) where

  type Tape (Convolution 'WithBias 'SameLower channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (BiasConvolution kernel bias _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        eb = extract bias
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters)
        cs = fromIntegral $ natVal (Proxy :: Proxy channels)

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padt = pady - div pady 2
        padl = padx - div padx 2

        out  = fromJust . create $ forwardBiasConv2d ex cs ix iy eb ek fs kx ky sx sy ix iy padl padt
    in  (S3D input, S3D out)

  runBackwards (BiasConvolution kernel _ _) (S3D input) (S3D dEdy) =     
    let ex = extract input
        ek = extract kernel
        ey = extract dEdy
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows )
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols )
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        fs = fromIntegral $ natVal (Proxy :: Proxy filters   )
        cs = fromIntegral $ natVal (Proxy :: Proxy channels  )

        pady = (ix - 1) * sx + kx - ix
        padx = (iy - 1) * sy + ky - iy

        padr = div padx 2
        padb = div pady 2
        padl = padx - padr
        padt = pady - padb

        (dx', dw', db') = backwardBiasConv2d ex cs ix iy ek fs kx ky sx sy padl padt padr padb ey ix iy
        dx              = fromJust . create $ dx'
        dw              = fromJust . create $ dw'
        db              = fromJust . create $ db'
    in  (BiasConvolution' dw db, S3D dx)


{---------------------------------}
{--    Other Layer instances    --}
{---------------------------------}

-- | A two dimentional image may have a convolution filter applied to it.
--   Case without bias vector
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithoutBias 'NoPadding 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (Convolution 'WithoutBias 'NoPadding 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithoutBias 'SameUpper 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) where
  type Tape (Convolution 'WithoutBias 'SameUpper 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithoutBias 'SameLower 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) where
  type Tape (Convolution 'WithoutBias 'SameLower 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a convolution filter applied to it.
--   Case with bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithBias 'NoPadding 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (Convolution 'WithBias 'NoPadding 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithBias 'SameUpper 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) where
  type Tape (Convolution 'WithBias 'SameUpper 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithBias 'SameLower 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) where
  type Tape (Convolution 'WithBias 'SameLower 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 inputRows inputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimensional image may have a convolution filter applied to it producing
--   a two dimensional image if both channels and filters is 1.
--   Case without bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , KnownNat outputRows
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithoutBias 'NoPadding 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (Convolution 'WithoutBias 'NoPadding 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithoutBias 'SameUpper 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithoutBias 'SameUpper 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithoutBias 'SameLower 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithoutBias 'SameLower 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

-- | A two dimensional image may have a convolution filter applied to it producing
--   a two dimensional image if both channels and filters is 1.
--   Case with bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , KnownNat outputRows
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithBias 'NoPadding 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (Convolution 'WithBias 'NoPadding 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithBias 'SameUpper 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithBias 'SameUpper 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         ) => Layer (Convolution 'WithBias 'SameLower 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithBias 'SameLower 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

-- | A three dimensional image can produce a 2D image from a convolution with 1 filter
--   Case without bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , KnownNat outputRows
         , KnownNat channels
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithoutBias 'NoPadding channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (Convolution 'WithoutBias 'NoPadding channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithoutBias 'SameUpper channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithoutBias 'SameUpper channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1))

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithoutBias 'SameLower channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithoutBias 'SameLower channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1))

-- | A three dimensional image can produce a 2D image from a convolution with 1 filter
--   Case with bias vector.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , KnownNat outputRows
         , KnownNat channels
         , OutputShapeIsOkay inputRows 0 kernelRows strideRows outputRows
         , OutputShapeIsOkay inputCols 0 kernelCols strideCols outputCols
         ) => Layer (Convolution 'WithBias 'NoPadding channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (Convolution 'WithBias 'NoPadding channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithBias 'SameUpper channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithBias 'SameUpper channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1))

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat channels
         ) => Layer (Convolution 'WithBias 'SameLower channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) where
  type Tape (Convolution 'WithBias 'SameLower channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 inputRows inputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 inputRows inputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 inputRows inputCols 1))

{--------------------}
{-- ONNX Instances --}
{--------------------}

instance OnnxOperator (Convolution 'WithBias padding channels filters kernelRows kernelCols strideRows strideCols) where
  onnxOpTypeNames _ = ["Conv"]

instance OnnxOperator (Convolution 'WithoutBias padding channels filters kernelRows kernelCols strideRows strideCols) where
  onnxOpTypeNames _ = ["Conv"]

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat channels
         , KnownNat (kernelRows * kernelCols * channels)
         ) => OnnxLoadable (Convolution 'WithoutBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols) where
  loadOnnxNode inits node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape
    hasCorrectPadding node (Proxy :: Proxy 0) (Proxy :: Proxy 0) (Proxy :: Proxy 0) (Proxy :: Proxy 0)

    case node ^. #input of
      [_, w] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        return (Convolution filterWeights mkListStore)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat channels
         , KnownNat (kernelRows * kernelCols * channels)
         ) => OnnxLoadable (Convolution 'WithoutBias 'SameUpper channels filters kernelRows kernelCols strideRows strideCols) where
  loadOnnxNode inits node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape

    case node ^. #input of
      [_, w] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        return (Convolution filterWeights mkListStore)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat channels
         , KnownNat padl, KnownNat padt, KnownNat padr, KnownNat padb
         ) => OnnxLoadable (Convolution 'WithoutBias ('Padding padl padt padr padb) channels filters kernelRows kernelCols strideRows strideCols) where
  loadOnnxNode inits node = do
    node `doesNotHaveAttribute` "auto_pad"

    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape

    hasCorrectPadding node (Proxy :: Proxy padl) (Proxy :: Proxy padr) (Proxy :: Proxy padt) (Proxy :: Proxy padb)

    case node ^. #input of
      [_, w] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        return (Convolution filterWeights mkListStore)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]

instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat channels
         , KnownNat (kernelRows * kernelCols * channels)
         ) => OnnxLoadable (Convolution 'WithBias padding channels filters kernelRows kernelCols strideRows strideCols) where
  loadOnnxNode inits node = do
    node & hasSupportedDilations
    node & hasSupportedGroup

    (node `hasMatchingShape` "kernel_shape") kernelShape
    (node `hasMatchingShape` "strides"     ) strideShape

    -- todo: proper checking to see if auto_pad attribute is valid

    case node ^. #input of
      [_, w, b] -> do
        filterWeights <- tr <$> readInitializerTensorIntoMatrix inits w
        filterBias    <- readInitializerVector inits b
        return (BiasConvolution filterWeights filterBias mkListStore)
      _ -> onnxIncorrectNumberOfInputs
      where
        kernelShape = [natVal (Proxy :: Proxy kernelRows), natVal (Proxy :: Proxy kernelCols)]
        strideShape = [natVal (Proxy :: Proxy strideRows), natVal (Proxy :: Proxy strideCols)]

{--------------------}
{-- Misc Instances --}
{--------------------}

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         ) =>
         LayerOptimizerData (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) where
  type MomentumExpOptResult (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  type MomentumDataType (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = konst 0

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         ) =>
         LayerOptimizerData (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) where
  type MomentumExpOptResult (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  type MomentumDataType (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  getData = undefined
  setData = undefined
  newData = undefined

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         ) => LayerOptimizerData (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) where
  type MomentumExpOptResult (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam)  = [L (kernelRows * kernelColumns * channels) filters]
  type MomentumDataType (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) = L (kernelRows * kernelColumns * channels) filters
  getData = getListStore
  setData = setListStore
  newData _ _ = konst 0

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => LayerOptimizerData (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) where
  type MomentumExpOptResult (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam)  = [L (kernelRows * kernelColumns * channels) filters]
  type MomentumDataType (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) = L (kernelRows * kernelColumns * channels) filters
  getData = undefined
  setData = undefined
  newData = undefined

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns) => FoldableGradient (Convolution' hasBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  mapGradient f (Convolution' kernelGradient) = Convolution' (dmmap f kernelGradient)
  mapGradient f (BiasConvolution' kernelGradient biasGradient) = BiasConvolution' (dmmap f kernelGradient) (dvmap f biasGradient)
  squaredSums (Convolution' kernelGradient) = [sumM . squareM $ kernelGradient]
  squaredSums (BiasConvolution' kernelGradient biasGradient) = [sumM . squareM $ kernelGradient, sumV . squareV $ biasGradient ]

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => Serialize (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Convolution w store) = do
    putListOf put . toList . flatten . extract $ w
    put (fmap (toList . flatten . extract) store)
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy filters)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      store <- fmap (fromMaybe (error "Vector of incorrect size") . create . reshape f . LA.fromList)  <$> get
      return $ Convolution wN store

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => Serialize (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (BiasConvolution w b store) = do
    putListOf put . toList . flatten . extract $ w
    putListOf put . toList . extract $ b
    put (fmap (toList . flatten . extract) store)
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy filters)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      wB    <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      store <- fmap (fromMaybe (error "Vector of incorrect size") . create . reshape f . LA.fromList)  <$> get
      return $ BiasConvolution wN wB store

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => Serialize (Convolution' 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Convolution' w) = do
    putListOf put . toList . flatten . extract $ w
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy filters)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      return $ Convolution' wN

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => Serialize (Convolution' 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (BiasConvolution' w b) = do
    putListOf put . toList . flatten . extract $ w
    putListOf put . toList . extract $ b
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy filters)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      wB    <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      return $ BiasConvolution' wN wB

instance NFData (Convolution hasBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  rnf (BiasConvolution a b c) = rnf a `seq` rnf b `seq` rnf c
  rnf (Convolution a b)       = rnf a `seq` rnf b

instance NFData (Convolution' hasBias padding channels filters kernelRows kernelColumns strideRows strideColumns) where
  rnf (BiasConvolution' a b) = rnf a `seq` rnf b
  rnf (Convolution' a)       = rnf a

instance Show (Convolution hasBias padding c f k k' s s') where
  show (BiasConvolution _ _ _) = "Bias Convolution"
  show (Convolution _ _)       = "Convolution"
