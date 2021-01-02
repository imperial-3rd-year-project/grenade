{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE AllowAmbiguousTypes   #-}
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

module Grenade.Layers.Transpose where

import           Control.DeepSeq                   (NFData (..))

import           Data.Kind                         (Type)
import           Data.Maybe                        (fromJust)
import           Data.Proxy
import           Data.Serialize
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra             as LA
import           Numeric.LinearAlgebra.Static      (R)
import qualified Numeric.LinearAlgebra.Static      as H

import           Grenade.Core
import           Grenade.Layers.Internal.Transpose
import           Grenade.Onnx

 -- todo: we can probably use a type family to represent this much better
data Transpose :: Nat
               -> Shape
               -> Shape
               -> Type where
  Transpose  :: ( KnownNat dimensions )
          => !(R dimensions)
          -> Transpose dimensions input output

instance Show (Transpose dimensions input output) where
  show (Transpose mat) = "Transpose " ++ show mat

instance UpdateLayer (Transpose dimensions input output) where
  type Gradient (Transpose dimensions input output) = ()
  runUpdate _ x _  = x
  reduceGradient _ = ()

instance ( KnownNat dimensions ) => RandomLayer (Transpose dimensions input output) where
  createRandomWith _ _ = pure initTranspose

initTranspose :: forall dimensions input output. ( KnownNat dimensions )
              => Transpose dimensions input output
initTranspose =
  let ds    = fromIntegral $ natVal (Proxy :: Proxy dimensions)
      perms = H.fromList [1..ds] :: R dimensions
  in Transpose perms

instance ( KnownNat dimensions ) => Serialize (Transpose dimensions input output) where
  put (Transpose perms) = putListOf put . LA.toList . H.extract $ perms
  get                   = do
    perms <- maybe (fail "Vector of incorrect size") return . H.create . LA.fromList =<< getListOf get
    return $ Transpose perms

instance ( KnownNat i, KnownNat j, KnownNat k, KnownNat l, KnownNat a, KnownNat b, KnownNat c, KnownNat d )
  => Layer (Transpose 4 ('D4 i j k l) ('D4 a b c d)) ('D4 i j k l) ('D4 a b c d) where

  type Tape (Transpose 4 ('D4 i j k l) ('D4 a b c d)) ('D4 i j k l) ('D4 a b c d) = ()

  runForwards (Transpose perms) (S4D x)
    = let n  = fromIntegral $ natVal (Proxy :: Proxy i)
          c  = fromIntegral $ natVal (Proxy :: Proxy j)
          h  = fromIntegral $ natVal (Proxy :: Proxy k)
          w  = fromIntegral $ natVal (Proxy :: Proxy l)
          x' = H.extract x
          perms' = H.extract perms
          r  = transpose4d [n, c, h, w] perms' x'
      in  ((), S4D . fromJust . H.create $ r)

  runBackwards = undefined

instance OnnxOperator (Transpose dimensions input output) where
  onnxOpTypeNames _ = ["Transpose"]

instance OnnxLoadable (Transpose 4 input output) where
  loadOnnxNode _ node = readIntsAttribute "perm" node >>= formatPerm
    where
      formatPerm :: [Int] -> Either OnnxLoadFailure (Transpose 4 input output)
      formatPerm [_, _, n, c, h, w]
        = let centered = map (\x -> fromIntegral $ x - 2) [n, c, h, w]
              perms :: R 4 = H.fromList centered
          in  return $ Transpose perms
      formatPerm _ = loadFailureReason "Permutation shape incorrect of Transpose"

instance NFData (Transpose dims input output) where
  rnf (Transpose perms) = rnf perms
