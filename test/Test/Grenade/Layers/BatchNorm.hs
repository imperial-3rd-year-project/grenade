{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.BatchNorm where

import           Grenade.Core.Layer
import           Grenade.Core.Shape
import           Grenade.Layers.BatchNormalisation
import           Grenade.Utils.ListStore
import           Grenade.Utils.LinearAlgebra

import           Numeric.LinearAlgebra.Data        as D hiding (R, (===))
import           Numeric.LinearAlgebra.Static      as H hiding ((===))

import           GHC.TypeLits
import           Hedgehog
import qualified Hedgehog.Gen                      as Gen
import qualified Hedgehog.Range                    as Range

import           Test.Hedgehog.Hmatrix

convToList = D.toList . H.extract

batchNormLayer :: Bool -> BatchNorm 1 1 3 90
batchNormLayer training = 
  let gamma        = H.fromList [1, 1, 1]                :: R 3
      beta         = H.fromList [0, 0, 0]                :: R 3
      running_mean = H.fromList [-0.143, -0.135, -0.126] :: R 3
      running_var  = H.fromList [1.554, 1.550, 1.574]    :: R 3
  in  BatchNorm training (BatchNormParams gamma beta) running_mean running_var mkListStore

prop_batchnorm_initialisation = withTests 1 $ property $ do
  let bn :: BatchNorm 1 1 5 90                                            = initBatchNorm
      BatchNorm t (BatchNormParams gamma beta) running_mean running_var _ = bn

      zeroes  = replicate 5 0
      ones    = replicate 5 1

  t      === True
  ones   === convToList gamma
  zeroes === convToList beta
  zeroes === convToList running_mean
  ones   === convToList running_var

prop_batchnorm_1D_training_forward_pass_tf = withTests 1 $ property $ do
  let bn = batchNormLayer True :: BatchNorm 1 1 3 90 

      batch1 = map (S1D . H.fromList) [[0.5, 0.1, 0.5], [0.3, 0.2, 0.3], [0.2, 0.01, 0.3]] :: [S ('D1 3)]
      batch2 = map (S1D . H.fromList) [[0.63733305, 0.36265903, 0.23532884], [0.32620134, 0.75205251, 0.94982708], [0.82240409, 0.41710665, 0.52376456], [0.88524989, 0.88508441, 0.92493869], [0.88527866, 0.74423246, 0.91648224]] :: [S ('D1 3)]

      (_, output1) = runBatchForwards bn batch1
      (_, output2) = runBatchForwards bn batch2

      -- these reference values were produced using Python and tf.nn.batch_normalization
      refOutput1 = map (S1D . H.fromList) [[ 1.33626326, -0.04295011,  1.41413402], [-0.26725265,  1.24555325, -0.70706701], [-1.06901061, -1.20260314, -0.70706701]] :: [S ('D1 3)]
      refOutput2 = map (S1D . H.fromList) [[-0.34738233, -1.31601869, -1.66528728], [-1.80872815,  0.58498266,  0.84102402], [ 0.52187266, -1.05020787, -0.65351473], [ 0.81705134,  1.23443837,  0.75372073], [ 0.81718647,  0.54680553,  0.72405726]] :: [S ('D1 3)]

  assert . and $ zipWith allClose output1 refOutput1
  assert . and $ zipWith allClose output2 refOutput2

prop_batchnorm_1D_testing_forward_pass_tf = withTests 1 $ property $ do
  let bn = batchNormLayer False :: BatchNorm 1 1 3 90 

      batch1 = map (S1D . H.fromList) [[0.5, 0.1, 0.5], [0.3, 0.2, 0.3], [0.2, 0.01, 0.3]] :: [S ('D1 3)]
      batch2 = map (S1D . H.fromList) [[0.63733305, 0.36265903, 0.23532884], [0.32620134, 0.75205251, 0.94982708], [0.82240409, 0.41710665, 0.52376456], [0.88524989, 0.88508441, 0.92493869], [0.88527866, 0.74423246, 0.91648224]] :: [S ('D1 3)]

      (_, output1) = runBatchForwards bn batch1
      (_, output2) = runBatchForwards bn batch2

      -- these reference values were produced using Python and tf.nn.batch_normalization
      refOutput1 = map (S1D . H.fromList) [[0.51580474, 0.18875648, 0.49896701], [0.3553678, 0.26907839, 0.33955263], [0.27514934, 0.11646677, 0.33955263]] :: [S ('D1 3)]
      refOutput2 = map (S1D . H.fromList) [[0.6259712, 0.39972922, 0.28800506], [0.37638612, 0.71249749, 0.85751153], [0.77443235, 0.44346259, 0.51790907], [0.82484629, 0.81935125, 0.8376737 ], [0.82486937, 0.70621628, 0.8309333 ]] :: [S ('D1 3)]

  assert . and $ zipWith allClose output1 refOutput1
  assert . and $ zipWith allClose output2 refOutput2

prop_batchnorm_1D_testing_forward_pass_paper = property $ do
  let bn = batchNormLayer True :: BatchNorm 1 1 3 90 
      BatchNorm _ (BatchNormParams γ β) _ _ _ = bn

  batch :: [S ('D1 3)] <- forAll . sequence $ replicate 5 genOfShape 

  let refys  = map S1D $ referenceRunFowards (map (\(S1D x) -> x) batch) γ β
      (_, ys) = runBatchForwards bn batch
  
  assert . and $ zipWith allClose ys refys

prop_batchnorm_1D_testing_backward_pass = property $ do
  let bn = batchNormLayer True :: BatchNorm 1 1 3 90 
      BatchNorm _ (BatchNormParams γ _) _ _ _ = bn

  sxs    :: [S ('D1 3)] <- forAll $ Gen.list (Range.singleton 32) genOfShape
  sdldys :: [S ('D1 3)] <- forAll $ Gen.list (Range.singleton 32) genOfShape

  let m     = 32 :: Double
      ε     = 0.000001 :: Double
      xs    = map (\(S1D x) -> x) sxs    :: [R 3]
      dldys = map (\(S1D x) -> x) sdldys :: [R 3]
      -- the following are taken directly from https://arxiv.org/pdf/1502.03167.pdf%20http://arxiv.org/abs/1502.03167.pdf
      -- Section 3
      μ     = dvmap (\x -> x / m) (sum xs) :: R 3
      σ²    = dvmap (\x -> x / m) $ sum $ map (\x -> (x - μ)**2) xs :: R 3
      std   = vsqrt $ dvmap (\a -> a + ε) σ²
      norm  = dvmap (\a -> 1 / a) std
      x̄s    = map (\x -> (x - μ) / std) xs

      dldx̄s = map (* γ) dldys
      dldσ² = dvmap (\a -> (-0.5) * a) $ sum $ zipWith (\dldx̄ x -> dldx̄ * (x - μ) * std ** (-3)) dldx̄s xs 
      dldμ  = (sum $ map (\dldx̄ -> dldx̄ * (-norm)) dldx̄s) + dldσ² * (dvmap (\x -> x / m) (sum $ map (\x -> (-2) * (x - μ)) xs))
      dldxs = zipWith (\dldx̄ x -> dldx̄ * norm + dldσ² * (dvmap (\a -> 2 * a / m) (x - μ)) + (dvmap (/m) dldμ)) dldx̄s xs
      dldγ  = sum $ zipWith (*) dldys x̄s 
      dldβ  = sum dldys

      t = TrainBatchNormTape x̄s std 0 0 :: BatchNormTape 1 1 3
      tape = t       :: Tape (BatchNorm 1 1 3 90) ('D1 3) ('D1 3)
      tape' = [tape] :: [Tape (BatchNorm 1 1 3 90) ('D1 3) ('D1 3)]
      ([BatchNormGrad _ _ gamma' beta'], dxs) = runBatchBackwards bn tape' sdldys :: ([BatchNormGrad 1 1 3], [S ('D1 3)])

  assert $ allCloseV dldγ gamma'
  assert $ allCloseV dldβ beta'
  assert . and $ zipWith allClose (map S1D dldxs) dxs

referenceRunFowards :: KnownNat n => [R n] -> R n -> R n -> [R n]
referenceRunFowards xs γ β
  = let m     = fromIntegral $ length xs :: Double
        ε     = 0.000001
        -- the following are taken directly from https://arxiv.org/pdf/1502.03167.pdf%20http://arxiv.org/abs/1502.03167.pdf
        -- Section 3
        μ     = dvmap (\x -> x / m) $ sum xs
        σ²    = dvmap (\x -> x / m) $ sum $ map (\x -> (x - μ)**2) xs
        std   = vsqrt $ dvmap (+ε) σ²
        x̄s    = map (\x -> (x - μ) / std) xs
        ys    = map (\x̄ -> γ * x̄ + β) x̄s
    in  ys

tests :: IO Bool
tests = checkParallel $$(discover)
