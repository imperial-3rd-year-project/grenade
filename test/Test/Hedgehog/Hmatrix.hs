{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Test.Hedgehog.Hmatrix where

import           GHC.Stack                    (HasCallStack,
                                               withFrozenCallStack)

import           Data.Singletons
import           Data.Singletons.TypeLits
import           Grenade

import           GHC.TypeLits

import           Hedgehog                     (Gen, MonadTest, diff)
import qualified Hedgehog.Gen                 as Gen
import qualified Hedgehog.Range               as Range

import           Numeric.LinearAlgebra        (norm_Inf)
import           Numeric.LinearAlgebra.Data   hiding ((===))
import qualified Numeric.LinearAlgebra.Static as H

import           Test.Hedgehog.Compat

randomVector :: forall n. ( KnownNat n ) =>  Gen (H.R n)
randomVector = (\s -> H.randomVector s H.Uniform * 2 - 1) <$> Gen.int Range.linearBounded

randomPositiveVector :: forall n. ( KnownNat n ) =>  Gen (H.R n)
randomPositiveVector = (\s -> H.randomVector s H.Uniform) <$> Gen.int Range.linearBounded

randomVectorNormalised :: forall n. ( KnownNat n ) =>  Gen (H.R n)
randomVectorNormalised = (\s -> sigmoid ((H.randomVector s H.Uniform) * 2 - 1)) <$> Gen.int Range.linearBounded
  where
    sigmoid :: Floating a => a -> a
    sigmoid x = 1/(1 + exp (-x))

uniformSample :: forall m n. ( KnownNat m, KnownNat n ) => Gen (H.L m n)
uniformSample = (\s -> H.uniformSample s (-1) 1 ) <$> Gen.int Range.linearBounded

-- | Generate random data of the desired shape
genOfShape :: forall x. ( SingI x ) => Gen (S x)
genOfShape =
  case (sing :: Sing x) of
    D1Sing l ->
      withKnownNat l $
        S1D <$> randomVector
    D2Sing r c ->
      withKnownNat r $ withKnownNat c $
        S2D <$> uniformSample
    D3Sing r c d ->
      withKnownNat r $ withKnownNat c $ withKnownNat d $
        S3D <$> uniformSample
    D4Sing n c h w ->
      withKnownNat n $ withKnownNat c $ withKnownNat h $ withKnownNat w $
        S4D <$> uniformSample

nice :: S shape -> String
nice (S1D x) = show . H.extract $ x
nice (S2D x) = show . H.extract $ x
nice (S3D x) = show . H.extract $ x
nice (S4D x) = show . H.extract $ x

allClose :: SingI shape => S shape -> S shape -> Bool
allClose xs ys = case xs - ys of
  (S1D x) -> H.norm_Inf x < 0.0001
  (S2D x) -> H.norm_Inf x < 0.0001
  (S3D x) -> H.norm_Inf x < 0.0001
  (S4D x) -> H.norm_Inf x < 0.0001

allCloseP :: SingI shape => S shape -> S shape -> RealNum -> Bool
allCloseP xs ys p = case xs - ys of
  (S1D x) -> H.norm_Inf x < p
  (S2D x) -> H.norm_Inf x < p
  (S3D x) -> H.norm_Inf x < p
  (S4D x) -> H.norm_Inf x < p

allCloseV :: KnownNat n => H.R n -> H.R n -> Bool
allCloseV xs ys = H.norm_Inf (xs - ys) < 0.0001

-- | generate a 2D list with random elements
genLists :: Int -> Int -> Gen [[RealNum]]
genLists height width = Gen.list (Range.singleton height) $ Gen.list (Range.singleton width) (genRealNum (Range.constant (-2.0) 2.0))

genLists3D :: Int -> Int -> Int -> Gen [[[RealNum]]]
genLists3D depth height width
  = Gen.list (Range.singleton depth) $
      Gen.list (Range.singleton height) $
        Gen.list (Range.singleton width)
          (genRealNum (Range.constant (-2.0) 2.0))

extractVec :: KnownNat n => S ('D1 n) -> [RealNum]
extractVec (S1D vec) = toList $ H.extract vec

extractMat :: (KnownNat a, KnownNat b) => S ('D2 a b) -> [[RealNum]]
extractMat (S2D mat) = toLists $ H.extract mat

extractMat3D :: (KnownNat a, KnownNat b, KnownNat c) => S ('D3 a b c) -> [[RealNum]]
extractMat3D (S3D mat) = toLists $ H.extract mat

extractMat4D :: (KnownNat a, KnownNat b, KnownNat c, KnownNat d, KnownNat (a * b * c)) => S ('D4 a b c d) -> [[RealNum]]
extractMat4D (S4D mat) = toLists $ H.extract mat

elementsEqual :: SingI shape => S shape -> Bool
elementsEqual m = case m of
  S1D x -> listSameElements . toList $ H.extract x
  S2D x -> listSameElements . concat . toLists $ H.extract x
  S3D x -> listSameElements . concat . toLists $ H.extract x
  S4D x -> listSameElements . concat . toLists $ H.extract x

listSameElements :: Eq a => [a] -> Bool
listSameElements []  = True
listSameElements [_] = True
listSameElements (x:x':xs)
  | x == x'   = listSameElements (x':xs)
  | otherwise = False

maxVal :: S shape -> RealNum
maxVal ( S1D x ) = norm_Inf x
maxVal ( S2D x ) = norm_Inf x
maxVal ( S3D x ) = norm_Inf x
maxVal ( S4D x ) = norm_Inf x

isSimilarMatrixTo :: (MonadTest m, HasCallStack) => Matrix RealNum -> Matrix RealNum -> m ()
isSimilarMatrixTo x y =
  withFrozenCallStack $
    diff x (\a b -> norm_Inf (a - b) < precision) y

isSimilarVectorTo :: (MonadTest m, HasCallStack) => Vector RealNum -> Vector RealNum -> m ()
isSimilarVectorTo x y =
  withFrozenCallStack $
    diff x (\a b -> norm_Inf (a - b) < precision) y
