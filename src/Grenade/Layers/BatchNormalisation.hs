{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
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

module Grenade.Layers.BatchNormalisation where

import           Control.DeepSeq
import           Data.Kind                         (Type)
import           Data.List                         (transpose, zipWith5)
import           Data.Maybe                        (fromJust)
import           Data.Proxy
import           Data.Serialize
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra             as LA
import           Numeric.LinearAlgebra.Static      hiding (Seed)

import           Grenade.Core
import           Grenade.Layers.Internal.BatchNorm
import           Grenade.Layers.Internal.Update
import           Grenade.Onnx
import           Grenade.Types
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

import           Lens.Micro

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
                     , KnownNat columns)
                     => [R (channels * rows * columns)]  -- xnorm
                     -> R channels    -- std
                     -> R channels    -- running mean
                     -> R channels    -- running variance
                     -> BatchNormTape channels rows columns

data BatchNorm :: Nat -- The number of channels of the tensor
               -> Nat -- The number of rows of the tensor
               -> Nat -- The number of columns of the tensor
               -> Nat -- momentum
               -> Type where
  BatchNorm :: ( KnownNat channels
               , KnownNat rows
               , KnownNat columns
               , KnownNat momentum)
            => Bool                                 -- is running training
            -> BatchNormParams channels             -- gamma and beta
            -> R channels                           -- running mean
            -> R channels                           -- running variance
            -> RealNum                              -- epsilon
            -> ListStore (BatchNormParams channels) -- momentum store
            -> BatchNorm channels rows columns momentum

data BatchNormParams :: Nat -> Type where
  BatchNormParams :: KnownNat channels
                  => R channels -- gamma
                  -> R channels -- beta
                  -> BatchNormParams channels

data BatchNormGrad :: Nat -- The number of channels of the tensor
                   -> Nat -- The number of rows of the tensor
                   -> Nat -- The number of columns of the tensor
                   -> Type where
  BatchNormGrad :: ( KnownNat channels
                   , KnownNat rows
                   , KnownNat columns)
            => R channels -- running mean
            -> R channels -- running variance
            -> R channels -- dgamna
            -> R channels -- dbeta
            -> BatchNormGrad channels rows columns


-- | NFData instances

instance NFData (BatchNormTape channels rows columns) where
  rnf TestBatchNormTape = ()
  rnf (TrainBatchNormTape xnorm std mean var) = rnf xnorm `seq` rnf std `seq` rnf mean `seq` rnf var

instance NFData (BatchNormParams flattenSize) where
  rnf (BatchNormParams gamma beta)
    = rnf gamma `seq` rnf beta

instance NFData (BatchNorm channels rows columns momentum) where
  rnf (BatchNorm training bnparams mean var eps store)
    = rnf training `seq` rnf bnparams `seq` rnf mean `seq` rnf var `seq` rnf eps `seq` rnf store

-- | Show instance

instance Show (BatchNorm channels rows columns momentum) where
  show _ = "Batch Normalization"

-- Serialize instances

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum)
  => Serialize (BatchNorm channels rows columns momentum) where

  put (BatchNorm training bnparams mean var ε store) = do
    put training
    put bnparams
    putListOf put . LA.toList . extract $ mean
    putListOf put . LA.toList . extract $ var
    put ε
    put store

  get = do
    training <- get
    bnparams <- get
    mean     <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
    var      <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
    ε        <- get
    store    <- get

    return $ BatchNorm training bnparams mean var ε store

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
  type MomentumStore (BatchNorm channels rows columns mom) = ListStore (BatchNormParams channels)

  reduceGradient = undefined

  runUpdate opt@OptSGD{} x@(BatchNorm training (BatchNormParams oldGamma oldBeta) _ _ ε store) (BatchNormGrad runningMean runningVar dGamma dBeta)
    = let BatchNormParams oldGammaMomentum oldBetaMomentum = getData opt x store
          VectorResultSGD newGamma newGammaMomentum        = descendVector opt (VectorValuesSGD oldGamma dGamma oldGammaMomentum)
          VectorResultSGD newBeta  newBetaMomentum         = descendVector opt (VectorValuesSGD oldBeta dBeta oldBetaMomentum)
          newStore                                         = setData opt x store (BatchNormParams newGammaMomentum newBetaMomentum)
      in  BatchNorm training (BatchNormParams newGamma newBeta) runningMean runningVar ε newStore

  runUpdate opt@OptAdam{} x@(BatchNorm training (BatchNormParams oldGamma oldBeta) _ _ ε store) (BatchNormGrad runningMean runningVar dGamma dBeta)
    = let [BatchNormParams oldMGamma oldMBeta, BatchNormParams oldVGamma oldVBeta] = getData opt x store
          VectorResultAdam newGamma newMGamma newVGamma                            = descendVector opt (VectorValuesAdam (getStep store) oldGamma dGamma oldMGamma oldVGamma)
          VectorResultAdam newBeta  newMBeta  newVBeta                             = descendVector opt (VectorValuesAdam (getStep store) oldBeta  dBeta  oldMBeta  oldVBeta)
          newStore                                                                 = setData opt x store [BatchNormParams newMGamma newMBeta, BatchNormParams newVGamma newVBeta]
      in  BatchNorm training (BatchNormParams newGamma newBeta) runningMean runningVar ε newStore

  runSettingsUpdate NetworkSettings{trainingActive=training} (BatchNorm _ bnparams mean var ε store) = BatchNorm training bnparams mean var ε store

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat mom)
  => LayerOptimizerData (BatchNorm channels rows columns mom) (Optimizer 'SGD) where

  type MomentumDataType (BatchNorm channels rows columns mom) (Optimizer 'SGD) = BatchNormParams channels
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = BatchNormParams (konst 0) (konst 0)

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat mom)
  => LayerOptimizerData (BatchNorm channels rows columns mom) (Optimizer 'Adam) where

  type MomentumDataType (BatchNorm channels rows columns mom) (Optimizer 'Adam) = BatchNormParams channels
  type MomentumExpOptResult (BatchNorm channels rows columns mom) (Optimizer 'Adam) = [BatchNormParams channels]
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
      zeroes = replicate ch 0
      ones   = replicate ch 1
      gamma  = vector ones   :: R channels
      beta   = vector zeroes :: R channels
      mean   = vector zeroes :: R channels
      var    = vector ones   :: R channels
      ε      = 0.00001
  in BatchNorm True (BatchNormParams gamma beta) mean var ε mkListStore

instance (KnownNat rows, KnownNat momentum)
  => Layer (BatchNorm 1 1 rows momentum) ('D1 rows) ('D1 rows) where

  type Tape (BatchNorm 1 1 rows momentum) ('D1 rows) ('D1 rows) = BatchNormTape 1 1 rows

  runForwards (BatchNorm True _ _ _ _ _) _
    = error "Cannot train use batch size of 1 with BatchNorm layer during training"

  runForwards (BatchNorm False (BatchNormParams gamma beta) runningMean runningVar ε _) (S1D x)
    = let gamma'       = extract gamma LA.! 0
          beta'        = extract beta LA.! 0
          runningMean' = extract runningMean LA.! 0
          runningVar'  = extract runningVar LA.! 0

          std = sqrt $ runningVar' + ε

          y = dvmap (\x -> ((x - runningMean') / std ) * gamma' + beta') x

      in (TestBatchNormTape, S1D y)

  runBatchForwards bn@(BatchNorm False _ _ _ _ _) xs
    = let outs = map (snd . runForwards bn) xs
      in ([TestBatchNormTape], outs)

  runBatchForwards (BatchNorm True (BatchNormParams gamma beta) runningMean runningVar ε _) xs
    = let [m]             = vectorToList runningMean :: [RealNum]
          [v]             = vectorToList runningVar  :: [RealNum]
          [g]             = vectorToList gamma       :: [RealNum]
          [b]             = vectorToList beta        :: [RealNum]
          mom             = (/ 100) $ fromIntegral $ natVal (Proxy :: Proxy momentum)

          xs'             = map extractV xs

          sample_mean     = batchNormMean xs'                 :: RealNum
          sample_var      = batchNormVariance xs'             :: RealNum

          m'              = mom * m + (1 - mom) * sample_mean :: RealNum
          v'              = mom * v + (1 - mom) * sample_var  :: RealNum
          std             = sqrt $ sample_var + ε             :: RealNum

          x_extracted     = map (\(S1D x) -> x) xs
          x_normalised    = map (dvmap (\a -> (a - sample_mean) / std)) x_extracted
          scaledShifted   = map (dvmap (\a -> g * a + b)) x_normalised
          out             = map S1D scaledShifted

          stdV            = listToVector [std] :: R 1
          runningMeanV    = listToVector [m']  :: R 1
          runningVarV     = listToVector [v']  :: R 1

      in ([TrainBatchNormTape x_normalised stdV runningMeanV runningVarV], out)

  runBatchBackwards (BatchNorm True (BatchNormParams _ _) _ _ _ _) _ _
    = undefined

  runBatchBackwards _ _ _
    = undefined

instance (KnownNat rows, KnownNat columns, KnownNat momentum)
  => Layer (BatchNorm 1 rows columns momentum) ('D2 rows columns) ('D2 rows columns) where

  type Tape (BatchNorm 1 rows columns momentum) ('D2 rows columns) ('D2 rows columns) = BatchNormTape 1 rows columns

  runForwards (BatchNorm True _ _ _ _ _) _
    = error "Cannot train use batch size of 1 with BatchNorm layer during training"

  runForwards (BatchNorm False (BatchNormParams gamma beta) runningMean runningVar ε _) (S2D x)
    = let rows     = fromIntegral $ natVal (Proxy :: Proxy rows)
          columns  = fromIntegral $ natVal (Proxy :: Proxy columns)

          gamma'       = extract gamma
          beta'        = extract beta
          runningMean' = extract runningMean
          runningVar'  = extract runningVar
          mat          = extract x

          y  = batchnorm 1 rows columns ε mat gamma' beta' runningMean' runningVar'
          y' = fromJust . create $ y

      in (TestBatchNormTape, S2D y')

  runBatchForwards bn@(BatchNorm False _ _ _ _ _) xs
    = let outs = map (snd . runForwards bn) xs
      in ([TestBatchNormTape], outs)

  runBatchForwards (BatchNorm True (BatchNormParams gamma beta) runningMean runningVar ε _) xs
    = let [m]             = vectorToList runningMean :: [RealNum]
          [v]             = vectorToList runningVar  :: [RealNum]
          [g]             = vectorToList gamma       :: [RealNum]
          [b]             = vectorToList beta        :: [RealNum]
          mom             = (/ 100) $ fromIntegral $ natVal (Proxy :: Proxy momentum)

          xs'             = map (sflatten . extractM2D) xs

          sample_mean     = batchNormMean xs'                 :: RealNum
          sample_var      = batchNormVariance xs'             :: RealNum

          m'              = mom * m + (1 - mom) * sample_mean :: RealNum
          v'              = mom * v + (1 - mom) * sample_var  :: RealNum
          std             = sqrt $ sample_var + ε             :: RealNum

          x_extracted     = map (\(S2D x) -> x) xs
          x_normalised    = map (dmmap (\a -> (a - sample_mean) / std)) x_extracted
          scaledShifted   = map (dmmap (\a -> g * a + b)) x_normalised
          out             = map S2D scaledShifted

          x_normalised'   = map sflatten x_normalised
          stdV            = listToVector [std] :: R 1
          runningMeanV    = listToVector [m']  :: R 1
          runningVarV     = listToVector [v']  :: R 1

      in ([TrainBatchNormTape x_normalised' stdV runningMeanV runningVarV], out)

  runBatchBackwards (BatchNorm True (BatchNormParams _ _) _ _ _ _) _ _
    = undefined

  runBatchBackwards _ _ _
    = undefined

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum)
  => Layer (BatchNorm channels rows columns momentum) ('D3 rows columns channels) ('D3 rows columns channels) where

  type Tape (BatchNorm channels rows columns momentum) ('D3 rows columns channels) ('D3 rows columns channels) = BatchNormTape channels rows columns

  runForwards (BatchNorm True _ _ _ _ _) _
    = error "Cannot train use batch size of 1 with BatchNorm layer during training"

  runForwards (BatchNorm False (BatchNormParams gamma beta) runningMean runningVar ε _) (S3D x)
    = let rows     = fromIntegral $ natVal (Proxy :: Proxy rows)
          columns  = fromIntegral $ natVal (Proxy :: Proxy columns)
          channels = fromIntegral $ natVal (Proxy :: Proxy channels)

          gamma'       = extract gamma
          beta'        = extract beta
          runningMean' = extract runningMean
          runningVar'  = extract runningVar
          mat          = extract x

          y  = batchnorm channels rows columns ε mat gamma' beta' runningMean' runningVar'
          y' = fromJust . create $ y

      in (TestBatchNormTape, S3D y')

  runBatchForwards bn@(BatchNorm False _ _ _ _ _) xs
    = let outs = map (snd . runForwards bn) xs
      in ([TestBatchNormTape], outs)

  runBatchForwards (BatchNorm True (BatchNormParams gamma beta) runningMean runningVar ε _) xs
    = let ms     = vectorToList runningMean
          vs     = vectorToList runningVar
          gs     = vectorToList gamma
          bs     = vectorToList beta

          cs     = map splitChannels xs :: [[S ('D2 rows columns)]]
          cs'    = transpose cs

          f c g b m v = let gs' = listToVector [g] :: R 1
                            bs' = listToVector [b] :: R 1
                            ms' = listToVector [m] :: R 1
                            vs' = listToVector [v] :: R 1
                            bn' = BatchNorm True (BatchNormParams gs' bs') ms' vs' ε undefined :: BatchNorm 1 rows columns momentum
                        in  runBatchForwards bn' c

          (tapes, outs) = unzip $ zipWith5 f cs' gs bs ms vs
          outs' = transpose outs
      in (combineTapes tapes, map combineChannels outs')
    where
      combineTapes :: [[BatchNormTape 1 i j]] -> [BatchNormTape k i j]
      combineTapes = undefined

  runBatchBackwards (BatchNorm True (BatchNormParams _ _) _ _ _ _) _ _
    = undefined

  runBatchBackwards _ _ _ = undefined

instance OnnxOperator (BatchNorm channels rows columns momentum) where
  onnxOpTypeNames _ = ["BatchNormalization"]

instance (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum) => OnnxLoadable (BatchNorm channels rows columns momentum) where
  loadOnnxNode inits node = case node ^. #input of
    [_, scale, b, mean, var] -> do
      epsilon     <- readFloatAttributeToRealNum "epsilon" node
      loadedScale <- readInitializerVector inits scale
      loadedB     <- readInitializerVector inits b
      loadedMean  <- readInitializerVector inits mean
      loadedVar   <- readInitializerVector inits var

      return $ BatchNorm False (BatchNormParams loadedScale loadedB) loadedMean loadedVar epsilon mkListStore
    _               -> onnxIncorrectNumberOfInputs


