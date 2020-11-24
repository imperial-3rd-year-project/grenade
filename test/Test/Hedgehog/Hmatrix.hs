{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module Test.Hedgehog.Hmatrix where

import           Grenade
import           Data.Singletons
import           Data.Singletons.TypeLits

import           Hedgehog (Gen)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import qualified Numeric.LinearAlgebra.Static as H
import           Numeric.LinearAlgebra.Data   hiding ((===))

randomVector :: forall n. ( KnownNat n ) =>  Gen (H.R n)
randomVector = (\s -> H.randomVector s H.Uniform * 2 - 1) <$> Gen.int Range.linearBounded

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

nice :: S shape -> String
nice (S1D x) = show . H.extract $ x
nice (S2D x) = show . H.extract $ x
nice (S3D x) = show . H.extract $ x

allClose :: SingI shape => S shape -> S shape -> Bool 
allClose xs ys = case xs - ys of
  (S1D x) -> H.norm_Inf x < 0.0001
  (S2D x) -> H.norm_Inf x < 0.0001
  (S3D x) -> H.norm_Inf x < 0.0001 

allCloseP :: SingI shape => S shape -> S shape -> Double -> Bool 
allCloseP xs ys p = case xs - ys of
  (S1D x) -> H.norm_Inf x < p
  (S2D x) -> H.norm_Inf x < p
  (S3D x) -> H.norm_Inf x < p

allCloseV :: KnownNat n => H.R n -> H.R n -> Bool 
allCloseV xs ys = H.norm_Inf (xs - ys) < 0.0001

-- | generate a 2D list with random elements
genLists :: Int -> Int -> Gen [[Double]]
genLists height width = Gen.list (Range.singleton height) $ Gen.list (Range.singleton width) (Gen.double (Range.constant (-2.0) 2.0))

extractVec :: KnownNat n => S ('D1 n) -> [Double]
extractVec (S1D vec) = toList $ H.extract vec

extractMat :: (KnownNat a, KnownNat b) => S ('D2 a b) -> [[Double]]
extractMat (S2D mat) = toLists $ H.extract mat
