{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications    #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Batch where

import           Grenade.Core.Shape
import           Grenade.Core.Network
import           Grenade.Core.Layer
import           Grenade.Layers.FullyConnected
import           Grenade.Layers.Convolution
import           Grenade.Utils.ListStore

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))
import           Numeric.LinearAlgebra.Static as H hiding ((===))
import           Numeric.LinearAlgebra.Data as D hiding ((===))

import           Hedgehog
import           GHC.TypeLits
import           Data.Proxy

import           Test.Hedgehog.Compat

type FFNetwork = Network '[ FullyConnected 3 5, FullyConnected 5 4 ] '[ 'D1 3, 'D1 5, 'D1 4 ]

prop_networkBatchFeedforward = property $ do
  let bias  :: H.R 5              = H.fromList [1..5]
      bias' :: H.R 4              = H.fromList [1..4]
      acts  :: H.L 5 3            = H.fromList [1..15]
      acts' :: H.L 4 5            = H.fromList [1..20]
      fc    :: FullyConnected 3 5 = FullyConnected (FullyConnected' bias  acts)  mkListStore
      fc'   :: FullyConnected 5 4 = FullyConnected (FullyConnected' bias' acts') mkListStore
      ins   :: [S ('D1 3)]        = [S1D (H.fromList [1, 2, 3]), S1D (H.fromList [4, 5, 6])]
      net   :: FFNetwork          = fc :~> fc' :~> NNil
      (_, outs :: [S ('D1 4)]) = runBatchForwards net ins
      outs' = map (\(S1D v) -> (D.toList . H.extract) v) outs
  outs' === [[986, 2312, 3638, 4964], [2336, 5462, 8588, 11714]]

extractFCGrads :: Gradients '[FullyConnected 3 5, FullyConnected 5 4] -> [(Vector Double, Matrix Double)]
extractFCGrads ((FullyConnected' wB wN) :/> ((FullyConnected' wB' wN') :/> GNil))
  = [(H.extract wB, H.extract wN), (H.extract wB', H.extract wN')]

prop_networkBackpropCalculatesGradients = property $ do
  let bias  :: H.R 5              = H.fromList [1..5]
      bias' :: H.R 4              = H.fromList [1..4]
      acts  :: H.L 5 3            = H.fromList [1..15]
      acts' :: H.L 4 5            = H.fromList [1..20]
      fc    :: FullyConnected 3 5 = FullyConnected (FullyConnected' bias acts) mkListStore
      fc'   :: FullyConnected 5 4 = FullyConnected (FullyConnected' bias' acts') mkListStore
      ins   :: [S ('D1 3)]        = [S1D (H.fromList [1, 2, 3]), S1D (H.fromList [4, 5, 6])]
      net   :: FFNetwork          = fc :~> fc' :~> NNil
      (tapes, outs :: [S ('D1 4)]) = runBatchForwards net ins
      (grads, vs   :: [S ('D1 3)]) = runBatchBackwards net tapes outs
      (grad,  v    :: S ('D1 3))   = runBackwards net (tapes!!0) (outs!!0)
      (grad', v'   :: S ('D1 3))   = runBackwards net (tapes!!1) (outs!!1)
      grads'                       = map extractFCGrads grads
      grads''                      = map extractFCGrads [grad, grad']
      vs'                          = map (\(S1D vec) -> (D.toList . H.extract) vec) vs
      vs''                         = map (\(S1D vec) -> (D.toList . H.extract) vec) [v, v']

  grads' === grads''
  vs'    === vs''

prop_networkAveragesGradients = property $ do
  let bias  :: H.R 5                 = H.fromList [1..5]
      bias' :: H.R 4                 = H.fromList [1..4]
      acts  :: H.L 5 3               = H.fromList [1..15]
      acts' :: H.L 4 5               = H.fromList [1..20]
      fc    :: FullyConnected 3 5    = FullyConnected (FullyConnected' bias acts) mkListStore
      fc'   :: FullyConnected 5 4    = FullyConnected (FullyConnected' bias' acts') mkListStore
      ins   :: [S ('D1 3)]           = [S1D (H.fromList [1, 2, 3]), S1D (H.fromList [4, 5, 6])]
      net   :: FFNetwork             = fc :~> fc' :~> NNil
      (tapes, outs :: [S ('D1 4)])   = runBatchForwards net ins
      (grads, _    :: [S ('D1 3)])   = runBatchBackwards net tapes outs
      (grad,  _    :: S ('D1 3))     = runBackwards net (tapes!!0) (outs!!0)
      (grad', _    :: S ('D1 3))     = runBackwards net (tapes!!1) (outs!!1)
      rgrad                          = extractFCGrads (reduceGradient @FFNetwork grads)
      [(wB, wN), (wB', wN')]         = extractFCGrads grad
      [(wB'', wN''), (wB''', wN''')] = extractFCGrads grad'
      rgrad'                         = (0.5 * (wB + wB''), 0.5 * (wN + wN''))
      rgrad''                        = (0.5 * (wB' + wB'''), 0.5 * (wN' + wN'''))

  rgrad === [rgrad', rgrad'']


prop_convolutionCalculatesOutputOfBatches = property $ do
  let weights :: H.L 25 1 = H.fromList [1..25]
      convLayer :: Convolution 1 1 5 5 2 2 = Convolution weights mkListStore
      ins :: [S ('D2 11 11)] = [S2D (H.fromList [1..121]), S2D (H.fromList [2..122])]
      (_, outs :: [S ('D2 4 4)]) = runBatchForwards convLayer ins
      outs' = map (\(S2D v) -> (concat . D.toLists . H.extract) v) outs
  (take 9 (concat outs')) === [10925, 11575, 12225, 12875, 18075, 18725, 19375, 20025, 25225]

unwrapGradConv :: ( KnownNat c
              , KnownNat f
              , KnownNat kR
              , KnownNat kC
              , KnownNat sR
              , KnownNat sC
              , KnownNat kF
              , kF ~ (kR * kC * c)) => Convolution' c f kR kC sR sC -> Matrix Double
unwrapGradConv (Convolution' mat) = H.extract mat

prop_convolutionBackprop = property $ do
  let weights :: H.L 25 1 = H.fromList [1..25]
      convLayer :: Convolution 1 1 5 5 2 2 = Convolution weights mkListStore
      ins :: [S ('D2 11 11)] = [S2D (H.fromList [1..121]), S2D (H.fromList [2..122])]
      (tapes, outs :: [S ('D2 4 4)]) = runBatchForwards convLayer ins
      (grads, vs   :: [S ('D2 11 11)]) = runBatchBackwards convLayer tapes outs
      (grad,  v    :: S ('D2 11 11))   = runBackwards convLayer (tapes!!0) (outs!!0)
      (grad', v'   :: S ('D2 11 11))   = runBackwards convLayer (tapes!!1) (outs!!1)
      grads'                           = map unwrapGradConv grads
      grads''                          = map unwrapGradConv [grad, grad']
      vs'                              = map (\(S2D u) -> (D.toLists . H.extract) u) vs
      vs''                             = map (\(S2D u) -> (D.toLists . H.extract) u) [v, v']

  grads' === grads''
  vs'    === vs''

prop_convolutionAveragesGradients = property $ do
  let weights :: H.L 25 1                  = H.fromList (concat (replicate 5 [1, 2, 3, 4, 5]))
      convLayer :: Convolution 1 1 5 5 2 2 = Convolution weights mkListStore
      ins :: [S ('D2 11 11)]               = [S2D (H.fromList [1..121]), S2D (H.fromList [2..122])]
      (tapes, outs :: [S ('D2 4 4)])       = runBatchForwards convLayer ins
      (grads, _    :: [S ('D2 11 11)])     = runBatchBackwards convLayer tapes outs
      (grad,  _    :: S ('D2 11 11))       = runBackwards convLayer (tapes!!0) (outs!!0)
      (grad', _    :: S ('D2 11 11))       = runBackwards convLayer (tapes!!1) (outs!!1)
      rgrad                                = unwrapGradConv (reduceGradient @(Convolution 1 1 5 5 2 2) grads)
      w                                    = unwrapGradConv grad
      w'                                   = unwrapGradConv grad'
      rgrad'                       = 0.5 * (w + w') 

  rgrad === rgrad'


prop_fullyConnectedCalculatesOutputOfBatches = property $ do
  let bias :: H.R 5 = H.fromList [1..5]
      acts :: H.L 5 3 = H.fromList [1..15]
      fc :: FullyConnected 3 5 = FullyConnected (FullyConnected' bias acts) mkListStore
      ins :: [S ('D1 3)] = [S1D (H.fromList [1, 2, 3]), S1D (H.fromList [4, 5, 6])]
      (_, outs :: [S ('D1 5)]) = runBatchForwards fc ins
      outs' = map (\(S1D v) -> (D.toList . H.extract) v) outs
  outs' === [[15, 34, 53, 72, 91], [33, 79, 125, 171, 217]]

prop_fullyConnectedBackprop = property $ do
  let bias :: H.R 5 = H.fromList [1..5]
      acts :: H.L 5 3 = H.fromList [1..15]
      fc :: FullyConnected 3 5 = FullyConnected (FullyConnected' bias acts) mkListStore
      ins :: [S ('D1 3)] = [S1D (H.fromList [1, 2, 3]), S1D (H.fromList [4, 5, 6])]
      (tapes, outs :: [S ('D1 5)]) = runBatchForwards fc ins
      (grads, vs   :: [S ('D1 3)]) = runBatchBackwards fc tapes outs
      (grad,  v    :: S ('D1 3))   = runBackwards fc (tapes!!0) (outs!!0)
      (grad', v'   :: S ('D1 3))   = runBackwards fc (tapes!!1) (outs!!1)
      unwrapGrad                   = \(FullyConnected' wB wN) -> (H.extract wB, H.extract wN)
      grads'                       = map unwrapGrad grads
      grads''                      = map unwrapGrad [grad, grad']
      vs'                          = map (\(S1D vec) -> (D.toList . H.extract) vec) vs
      vs''                         = map (\(S1D vec) -> (D.toList . H.extract) vec) [v, v']

  grads' === grads''
  vs'    === vs''

prop_fullyConnectedAveragesGradients = property $ do
  let bias :: H.R 5 = H.fromList [1..5]
      acts :: H.L 5 3 = H.fromList [1..15]
      fc :: FullyConnected 3 5 = FullyConnected (FullyConnected' bias acts) mkListStore
      ins :: [S ('D1 3)] = [S1D (H.fromList [1, 2, 3]), S1D (H.fromList [4, 5, 6])]
      (tapes, outs :: [S ('D1 5)]) = runBatchForwards fc ins
      (grads, _    :: [S ('D1 3)]) = runBatchBackwards fc tapes outs
      (grad,  _    :: S ('D1 3))   = runBackwards fc (tapes!!0) (outs!!0)
      (grad', _    :: S ('D1 3))   = runBackwards fc (tapes!!1) (outs!!1)
      f                            = \(FullyConnected' bs ns) -> (H.extract bs, H.extract ns)
      rgrad                        = f (reduceGradient @(FullyConnected 3 5) grads)
      (wB, wN)                     = f grad
      (wB', wN')                   = f grad'
      rgrad'                       = (0.5 * (wB + wB'), 0.5 * (wN + wN'))

  rgrad === rgrad'
   
tests :: IO Bool
tests = checkParallel $$(discover)
