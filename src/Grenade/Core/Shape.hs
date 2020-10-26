{-# LANGUAGE CPP                  #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE StandaloneDeriving   #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE UndecidableInstances #-}
{-|
Module      : Grenade.Core.Shape
Description : Dependently typed shapes of data which are passed between layers of a network
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental


-}
module Grenade.Core.Shape (
    S (..)
  , Shape (..)
#if MIN_VERSION_singletons(2,6,0)
  , SShape (..)
#else
  , Sing (..)
#endif

  , randomOfShape
  , fromStorable
  , nk
  ) where

#if MIN_VERSION_singletons(2,6,0)
import           Data.Kind                    (Type)
#endif

import           Control.DeepSeq              (NFData (..))
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.Vector.Storable         (Vector)
import qualified Data.Vector.Storable         as V
import           GHC.TypeLits                 hiding (natVal)
import qualified Numeric.LinearAlgebra        as NLA
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra.Static as H
import           System.Random.MWC

import           Grenade.Types

-- | The current shapes we accept.
--   at the moment this is just one, two, and three dimensional
--   Vectors/Matricies.
--
--   These are only used with DataKinds, as Kind `Shape`, with Types 'D1, 'D2, 'D3.
data Shape
  = D1 Nat
  -- ^ One dimensional vector
  | D2 Nat Nat
  -- ^ Two dimensional matrix. Row, Column.
  | D3 Nat Nat Nat
  -- ^ Three dimensional matrix. Row, Column, Channels.
  | B1 Nat Nat 
  -- ^ Batch of one dimensional vectors. Depth, Row
  | B2 Nat Nat Nat
  -- ^ Batch of two dimensional vectors. Depth, Row, Column
  | B3 Nat Nat Nat Nat 
  -- ^ Batch of three dimensional vectors. Depth, Row, Column, Channels

-- | Concrete data structures for a Shape.
--
--   All shapes are held in contiguous memory.
--   3D is held in a matrix (usually row oriented) which has height depth * rows.
data S (n :: Shape) where
  S1D :: ( KnownNat len )
      => R len
      -> S ('D1 len)

  S2D :: ( KnownNat rows, KnownNat columns )
      => L rows columns
      -> S ('D2 rows columns)

  S3D :: ( KnownNat rows
         , KnownNat columns
         , KnownNat depth
         , KnownNat (rows * depth))
      => L (rows * depth) columns
      -> S ('D3 rows columns depth)
  
  S1B :: ( KnownNat depth, KnownNat rows )
      => L depth rows 
      -> S ('B1 depth rows)
  
  S2B :: ( KnownNat depth
         , KnownNat rows
         , KnownNat columns
         , KnownNat (rows * columns) )
      => L depth (rows * columns)
      -> S ('B2 depth rows columns)
  
  S3B :: ( KnownNat depth
         , KnownNat rows
         , KnownNat columns
         , KnownNat channels
         , KnownNat (rows * columns * channels) )
      => L depth (rows * columns * channels)
      -> S ('B3 depth rows columns channels)

deriving instance Show (S n)

-- Singleton instances.
--
-- These could probably be derived with template haskell, but this seems
-- clear and makes adding the KnownNat constraints simple.
-- We can also keep our code TH free, which is great.
#if MIN_VERSION_singletons(2,6,0)
-- In singletons 2.6 Sing switched from a data family to a type family.
type instance Sing = SShape

data SShape :: Shape -> Type where
  D1Sing :: Sing a -> SShape ('D1 a)
  D2Sing :: Sing a -> Sing b -> SShape ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> SShape ('D3 a b c)
  B1Sing :: Sing d -> Sing a -> SShape ('B1 d a)
  B2Sing :: KnownNat (a * b) => Sing d -> Sing a -> Sing b -> SShape ('B2 d a b) 
  B3Sing :: KnownNat (a * b * c) => Sing d -> Sing a -> Sing b -> Sing c -> SShape ('B3 d a b c)
#else
data instance Sing (n :: Shape) where
  D1Sing :: Sing a -> Sing ('D1 a)
  D2Sing :: Sing a -> Sing b -> Sing ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> Sing ('D3 a b c)
  B1Sing :: Sing d -> Sing a -> Sing ('B1 d a)
  B2Sing :: KnownNat (a * b) => Sing d -> Sing a -> Sing b -> Sing ('B2 d a b) 
  B3Sing :: KnownNat (a * b * c) => Sing d -> Sing a -> Sing b -> Sing c -> Sing ('B3 d a b c)
#endif

instance KnownNat a => SingI ('D1 a) where
  sing = D1Sing sing
instance (KnownNat a, KnownNat b) => SingI ('D2 a b) where
  sing = D2Sing sing sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat (a * c)) => SingI ('D3 a b c) where
  sing = D3Sing sing sing sing
instance (KnownNat d, KnownNat a) => SingI ('B1 d a) where
  sing = B1Sing sing sing 
instance (KnownNat d, KnownNat a, KnownNat b, KnownNat (a * b)) => SingI ('B2 d a b) where
  sing = B2Sing sing sing sing 
instance (KnownNat d, KnownNat a, KnownNat b, KnownNat c, KnownNat (a * b * c)) => SingI ('B3 d a b c) where
  sing = B3Sing sing sing sing sing

instance SingI x => Num (S x) where
  (+) = n2 (+)
  (-) = n2 (-)
  (*) = n2 (*)
  abs = n1 abs
  signum = n1 signum
  fromInteger x = nk (fromInteger x)

instance SingI x => Fractional (S x) where
  (/) = n2 (/)
  recip = n1 recip
  fromRational x = nk (fromRational x)

instance SingI x => Floating (S x) where
  pi = nk pi
  exp = n1 exp
  log = n1 log
  sqrt = n1 sqrt
  (**) = n2 (**)
  logBase = n2 logBase
  sin = n1 sin
  cos = n1 cos
  tan = n1 tan
  asin = n1 asin
  acos = n1 acos
  atan = n1 atan
  sinh = n1 sinh
  cosh = n1 cosh
  tanh = n1 tanh
  asinh = n1 asinh
  acosh = n1 acosh
  atanh = n1 atanh

--
-- I haven't made shapes strict, as sometimes they're not needed
-- (the last input gradient back for instance)
--
instance NFData (S x) where
  rnf (S1D x) = rnf x
  rnf (S2D x) = rnf x
  rnf (S3D x) = rnf x
  rnf (S1B x) = rnf x
  rnf (S2B x) = rnf x
  rnf (S3B x) = rnf x

-- | Generate random data of the desired shape
randomOfShape :: forall x . (SingI x) => IO (S x)
randomOfShape = do
  seed :: Int <- withSystemRandom . asGenST $ \gen -> uniform gen
  return $ case (sing :: Sing x) of
    D1Sing SNat ->
        S1D (randomVector seed Uniform * 2 - 1)

    D2Sing SNat SNat ->
        S2D (uniformSample seed (-1) 1)

    D3Sing SNat SNat SNat ->
        S3D (uniformSample seed (-1) 1)
      
    B1Sing SNat SNat ->
        S1B (uniformSample seed (-1) 1)

    B2Sing SNat SNat SNat ->
        S2B (uniformSample seed (-1) 1)

    B3Sing SNat SNat SNat SNat ->
        S3B (uniformSample seed (-1) 1)

-- | Generate a shape from a Storable Vector.
--
--   Returns Nothing if the vector is of the wrong size.
fromStorable :: forall x. SingI x => Vector RealNum -> Maybe (S x)
fromStorable xs = case sing :: Sing x of
    D1Sing SNat ->
      S1D <$> H.create xs

    D2Sing SNat SNat ->
      S2D <$> mkL xs

    D3Sing SNat SNat SNat ->
      S3D <$> mkL xs

    B1Sing SNat SNat ->
      S1B <$> mkL xs 

    B2Sing SNat SNat SNat ->
      S2B <$> mkL xs 

    B3Sing SNat SNat SNat SNat ->
      S3B <$> mkL xs 
  where
    mkL :: forall rows columns. (KnownNat rows, KnownNat columns)
        => Vector RealNum -> Maybe (L rows columns)
    mkL v =
      let rows    = fromIntegral $ natVal (Proxy :: Proxy rows)
          columns = fromIntegral $ natVal (Proxy :: Proxy columns)
      in  if rows * columns == V.length v
             then H.create $ NLA.reshape columns v
             else Nothing


instance SingI x => Serialize (S x) where
  put i = (case i of
            (S1D x) -> putListOf put . NLA.toList . H.extract $ x
            (S2D x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
            (S3D x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
            (S1B x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
            (S2B x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
            (S3B x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
          ) :: PutM ()

  get = do
    Just i <- fromStorable . V.fromList <$> getListOf get
    return i

-- Helper function for creating the number instances
n1 :: ( forall a. Floating a => a -> a ) -> S x -> S x
n1 f (S1D x) = S1D (f x)
n1 f (S2D x) = S2D (f x)
n1 f (S3D x) = S3D (f x)
n1 f (S1B x) = S1B (f x)
n1 f (S2B x) = S2B (f x)
n1 f (S3B x) = S3B (f x)

-- Helper function for creating the number instances
n2 :: ( forall a. Floating a => a -> a -> a ) -> S x -> S x -> S x
n2 f (S1D x) (S1D y) = S1D (f x y)
n2 f (S2D x) (S2D y) = S2D (f x y)
n2 f (S3D x) (S3D y) = S3D (f x y)
n2 f (S1B x) (S1B y) = S1B (f x y)
n2 f (S2B x) (S2B y) = S2B (f x y)
n2 f (S3B x) (S3B y) = S3B (f x y)

-- Helper function for creating the number instances
nk :: forall x. SingI x => RealNum -> S x
nk x = case (sing :: Sing x) of
  D1Sing SNat ->
    S1D (konst x)

  D2Sing SNat SNat ->
    S2D (konst x)

  D3Sing SNat SNat SNat ->
    S3D (konst x)
  
  B1Sing SNat SNat -> 
    S1B (konst x)
  
  B2Sing SNat SNat SNat ->
    S2B (konst x)
  
  B3Sing SNat SNat SNat SNat ->
    S3B (konst x)
