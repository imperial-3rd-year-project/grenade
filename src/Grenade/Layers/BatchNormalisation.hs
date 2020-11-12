{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}

module Grenade.Layers.BatchNormalisation where

import           Control.DeepSeq
import           Data.Kind                      (Type)
import           Data.Maybe                     (fromJust)
import           Data.Proxy
import           Data.Serialize
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static   hiding (Seed)
import           Numeric.LinearAlgebra.Static   as H

import           Grenade.Core
import           Grenade.Layers.Internal.Update
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

data BatchNormTape :: Nat -- The number of channels of the tensor
                   -> Nat -- The number of rows of the tensor
                   -> Nat -- The number of columns of the tensor
                   -> Type where
  TestBatchNormTape  :: ( KnownNat channels
                     , KnownNat rows
                     , KnownNat columns)
                     => BatchNormTape channels rows columns

  TrainBatchNormTape :: ( KnownNat channels
                     , KnownNat rows
                     , KnownNat columns
                     , KnownNat flattenSize
                     , flattenSize ~ (channels * rows * columns))
                     => [R flattenSize]  -- xnorm
                     -> R flattenSize    -- std
                     -> R flattenSize    -- running mean
                     -> R flattenSize    -- running variance
                     -> BatchNormTape channels rows columns

data BatchNorm :: Nat -- The number of channels of the tensor
               -> Nat -- The number of rows of the tensor
               -> Nat -- The number of columns of the tensor
               -> Nat -- momentum
               -> Type where
  BatchNorm :: ( KnownNat channels
               , KnownNat rows
               , KnownNat columns
               , KnownNat momentum
               , KnownNat flattenSize
               , flattenSize ~ (channels * rows * columns))
            => Bool                                    -- is running training
            -> BatchNormParams flattenSize             -- gamma and beta
            -> R flattenSize                           -- running mean
            -> R flattenSize                           -- running variance
            -> ListStore (BatchNormParams flattenSize) -- momentum store
            -> BatchNorm channels rows columns momentum

data BatchNormParams :: Nat -> Type where
  BatchNormParams :: KnownNat flattenSize
                  => R flattenSize -- gamma
                  -> R flattenSize -- beta
                  -> BatchNormParams flattenSize

data BatchNormGrad :: Nat -- The number of channels of the tensor
                   -> Nat -- The number of rows of the tensor
                   -> Nat -- The number of columns of the tensor
                   -> Type where
  BatchNormGrad :: ( KnownNat channels
                   , KnownNat rows
                   , KnownNat columns
                   , KnownNat flattenSize
                   , flattenSize ~ (channels * rows * columns))
            => R flattenSize -- running mean
            -> R flattenSize -- running variance
            -> R flattenSize -- dgamna
            -> R flattenSize -- dbeta
            -> BatchNormGrad channels rows columns


-- | NFData instances

instance NFData (BatchNormTape channels rows columns) where
  rnf TestBatchNormTape = ()
  rnf (TrainBatchNormTape xnorm std mean var) = rnf xnorm `seq` rnf std `seq` rnf mean `seq` rnf var

instance NFData (BatchNormParams flattenSize) where
  rnf (BatchNormParams gamma beta)
    = rnf gamma `seq` rnf beta

instance NFData (BatchNorm channels rows columns momentum) where
  rnf (BatchNorm training bnparams mean var store)
    = rnf training `seq` rnf bnparams `seq` rnf mean `seq` rnf var `seq` rnf store

-- | Show instance

instance Show (BatchNorm channels rows columns momentum) where
  show (BatchNorm _ _ _ _ _) = "Batch Normalization"

-- Serialize instances

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum)
  => Serialize (BatchNorm channels rows columns momentum) where

  put (BatchNorm training bnparams mean var store) = do
    put training
    put bnparams
    putListOf put . LA.toList . extract $ mean
    putListOf put . LA.toList . extract $ var
    put store

  get = do
    let ch = fromIntegral $ natVal (Proxy :: Proxy channels)
    let r  = fromIntegral $ natVal (Proxy :: Proxy rows)
    let co = fromIntegral $ natVal (Proxy :: Proxy columns)

    training <- get
    bnparams <- get
    mean     <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
    var      <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
    store    <- get

    return $ BatchNorm training bnparams mean var store

instance KnownNat flattenSize => Serialize (BatchNormParams flattenSize) where
  put (BatchNormParams gamma beta) = do
    putListOf put . LA.toList . extract $ gamma
    putListOf put . LA.toList . extract $ beta

  get = do
    gamma <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
    beta  <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get

    return $ BatchNormParams gamma beta

-- | Neural network operations

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat mom, KnownNat (channels * rows * columns))
  => UpdateLayer (BatchNorm channels rows columns mom) where

  type Gradient (BatchNorm channels rows columns mom) = BatchNormGrad channels rows columns
  type MomentumStore (BatchNorm channels rows columns mom) = ListStore (BatchNormParams (channels * rows * columns))

  reduceGradient = head

  runUpdate opt@OptSGD{} x@(BatchNorm training (BatchNormParams oldGamma oldBeta) _ _ store) (BatchNormGrad runningMean runningVar dGamma dBeta)
    = let BatchNormParams oldGammaMomentum oldBetaMomentum = getData opt x store
          VectorResultSGD newGamma newGammaMomentum        = descendVector opt (VectorValuesSGD oldGamma dGamma oldGammaMomentum)
          VectorResultSGD newBeta  newBetaMomentum         = descendVector opt (VectorValuesSGD oldBeta dBeta oldBetaMomentum)
          newStore                                         = setData opt x store (BatchNormParams newGammaMomentum newBetaMomentum)
      in  BatchNorm training (BatchNormParams newGamma newBeta) runningMean runningVar newStore

  runUpdate opt@OptAdam{} x@(BatchNorm training (BatchNormParams oldGamma oldBeta) _ _ store) (BatchNormGrad runningMean runningVar dGamma dBeta)
    = let [BatchNormParams oldMGamma oldMBeta, BatchNormParams oldVGamma oldVBeta] = getData opt x store
          VectorResultAdam newGamma newMGamma newVGamma                            = descendVector opt (VectorValuesAdam (getStep store) oldGamma dGamma oldMGamma oldVGamma)
          VectorResultAdam newBeta  newMBeta  newVBeta                             = descendVector opt (VectorValuesAdam (getStep store) oldBeta  dBeta  oldMBeta  oldVBeta)
          newStore                                                                 = setData opt x store [BatchNormParams newMGamma newMBeta, BatchNormParams newVGamma newVBeta]
      in  BatchNorm training (BatchNormParams newGamma newBeta) runningMean runningVar newStore

  runSettingsUpdate NetworkSettings{trainingActive=training} (BatchNorm _ bnparams mean var store) = BatchNorm training bnparams mean var store

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat mom, KnownNat flattenSize, flattenSize ~ (channels * rows * columns))
  => LayerOptimizerData (BatchNorm channels rows columns mom) (Optimizer 'SGD) where

  type MomentumDataType (BatchNorm channels rows columns mom) (Optimizer 'SGD) = BatchNormParams (channels * rows * columns)
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = BatchNormParams (konst 0) (konst 0)

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat mom, KnownNat flattenSize, flattenSize ~ (channels * rows * columns))
  => LayerOptimizerData (BatchNorm channels rows columns mom) (Optimizer 'Adam) where

  type MomentumDataType (BatchNorm channels rows columns mom) (Optimizer 'Adam) = BatchNormParams (channels * rows * columns)
  type MomentumExpOptResult (BatchNorm channels rows columns mom) (Optimizer 'Adam) = [BatchNormParams (channels * rows * columns)]
  getData     = getListStore
  setData     = setListStore
  newData _ _ = BatchNormParams (konst 0) (konst 0)

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum)
  => RandomLayer (BatchNorm channels rows columns momentum) where
  createRandomWith _ _ = pure initBatchNorm

initBatchNorm :: forall channels rows columns momentum.
  (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum)
  => BatchNorm channels rows columns momentum
initBatchNorm =
  let ch     = fromIntegral $ natVal (Proxy :: Proxy channels)
      r      = fromIntegral $ natVal (Proxy :: Proxy rows)
      co     = fromIntegral $ natVal (Proxy :: Proxy columns)
      zeroes = replicate (r * ch * co) 0
      ones   = replicate (r * ch * co) 1
      gamma  = vector ones   :: R (channels * rows * columns)
      beta   = vector zeroes :: R (channels * rows * columns)
      mean   = vector zeroes :: R (channels * rows * columns)
      var    = vector ones   :: R (channels * rows * columns)
  in BatchNorm True (BatchNormParams gamma beta) mean var mkListStore

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum, KnownNat i, (rows * columns) ~ i, (channels * rows * columns) ~ i)
  => Layer (BatchNorm channels rows columns momentum) ('D1 i) ('D1 i) where

  type Tape (BatchNorm channels rows columns momentum) ('D1 i) ('D1 i) = BatchNormTape channels rows columns

  runForwards (BatchNorm True _ _ _ _) _
    = error "Cannot train use batch size of 1 with BatchNorm layer during training"

  runForwards (BatchNorm False (BatchNormParams gamma beta) runningMean runningVar _) (S1D x)
    = let ε      = 0.000001
          std    = vsqrt $ dvmap (+ε) runningVar
          x_norm = (x - runningMean) / std
          out    = gamma * x_norm + beta
      in (TestBatchNormTape, S1D out)

  runBatchForwards bn@(BatchNorm False _ _ _ _) xs
    = let outs = map (snd . runForwards bn) xs
      in ([TestBatchNormTape], outs)

  runBatchForwards (BatchNorm True (BatchNormParams gamma beta) runningMean runningVar _) xs
    = let ε               = 0.000001
          m               = fromIntegral $ natVal (Proxy :: Proxy momentum)
          S1D sample_mean = bmean xs
          S1D sample_var  = bvar' (S1D sample_mean) xs
          running_mean'   = m * runningMean + (1 - m) * sample_mean
          running_var'    = m * runningVar + (1 - m) * sample_var
          std             = vsqrt $ dvmap (+ε) sample_var
          x_centered      = map (\(S1D x) -> x - sample_mean) xs
          x_norm          = map (/ std) x_centered
          out             = map (\x -> S1D $ gamma * x + beta) x_norm
      in ([TrainBatchNormTape x_norm std running_mean' running_var'], out)

  runBatchBackwards (BatchNorm True (BatchNormParams gamma _) _ _ _) [TrainBatchNormTape x_norm std running_mean' running_var'] douts
    = let douts'      = map (\(S1D v) -> v) douts
          dgamma_int  = zipWith (*) douts' x_norm
          dgamma      = sum dgamma_int
          dbeta       = sum douts'
          dx_norm     = map (* gamma) douts' :: [R i]

          n           = fromIntegral $ length douts :: Double
          dx_sum      = sum $ zipWith (*) x_norm dx_norm :: R i
          dx_sum'     = map (* dx_sum) x_norm :: [R i]
          dx_norm_sum = sum dx_norm :: R i
          dx_norm_N   = map (vscale n) dx_norm :: [R i]
          std'        = dvmap (\x -> 1 / (n * x)) std :: R i
          dx          = map (std' *) (zipWith (-) (map (\x -> x - dx_norm_sum) dx_norm_N) dx_sum')
      in ([BatchNormGrad running_mean' running_var' dgamma dbeta], map S1D dx)

instance (KnownNat (i * j), KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum, KnownNat i, KnownNat j, (channels * rows * columns) ~ (i * j), (i * j) ~ flattenSize, rows ~ i, columns ~ j)
  => Layer (BatchNorm channels rows columns momentum) ('D2 i j) ('D2 i j) where

  type Tape (BatchNorm channels rows columns momentum) ('D2 i j) ('D2 i j) = BatchNormTape channels rows columns

  runForwards (BatchNorm True _ _ _ _) _
    = error "Cannot train use batch size of 1 with BatchNorm layer during training"

  runForwards bn@(BatchNorm False (BatchNormParams gamma beta) runningMean runningVar _) (S2D x')
    = let ε      = 0.000001
          x      = sflatten x' :: R (i * j)
          std    = vsqrt $ dvmap (+ε) runningVar
          x_norm = (x - runningMean) / std
          out    = gamma * x_norm + beta
      in (TestBatchNormTape, S2D (sreshape out))

  runBatchForwards bn@(BatchNorm False _ _ _ _) xs
    = let outs = map (snd . runForwards bn) xs
      in ([TestBatchNormTape], outs)

  runBatchForwards (BatchNorm True (BatchNormParams gamma beta) runningMean runningVar _) xs
    = let ε                = 0.000001
          m                = fromIntegral $ natVal (Proxy :: Proxy momentum)
          S2D sample_mean' = bmean xs
          S2D sample_var'  = bvar' (S2D sample_mean') xs
          sample_mean      = sflatten sample_mean'
          sample_var       = sflatten sample_var'
          running_mean'    = m * runningMean + (1 - m) * sample_mean
          running_var'     = m * runningVar + (1 - m) * sample_var
          std              = vsqrt $ dvmap (+ε) sample_var :: R (i * j)
          x_centered       = map (\(S2D x) -> sflatten (x - sample_mean')) xs :: [R (i * j)]
          x_norm           = map (/ std) x_centered
          out              = map (\x -> S2D $ sreshape $ gamma * x + beta) x_norm :: [S ('D2 i j)]
      in ([TrainBatchNormTape x_norm std running_mean' running_var'], out)

  runBatchBackwards (BatchNorm True (BatchNormParams gamma _) _ _ _) [TrainBatchNormTape x_norm std running_mean' running_var'] douts
    = let douts'      = map (\(S2D v) -> sflatten v) douts :: [R (i * j)]
          dgamma_int  = zipWith (*) douts' x_norm
          dgamma      = sum dgamma_int
          dbeta       = sum douts'
          dx_norm     = map (* gamma) douts' -- :: [R i]

          n           = fromIntegral $ length douts :: Double
          dx_sum      = sum $ zipWith (*) x_norm dx_norm -- :: R i
          dx_sum'     = map (* dx_sum) x_norm -- :: [R i]
          dx_norm_sum = sum dx_norm -- :: R i
          dx_norm_N   = map (vscale n) dx_norm -- :: [R i]
          std'        = dvmap (\x -> 1 / (n * x)) std -- :: R i
          dx          = map (std' *) (zipWith (-) (map (\x -> x - dx_norm_sum) dx_norm_N) dx_sum')
          dx'         = map (fromJust . fromStorable . H.extract) dx :: [S ('D2 i j)]
      in ([BatchNormGrad running_mean' running_var' dgamma dbeta], dx')
