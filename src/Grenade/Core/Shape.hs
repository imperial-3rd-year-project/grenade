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
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-|
Module      : Grenade.Core.Shape
Description : Dependently typed shapes of data which are passed between layers of a network
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental


-}
module Grenade.Core.Shape (
    S (..)
  , T (..)
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
import           Data.Maybe                   (fromJust)
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.Vector.Storable         (Vector)
import qualified Data.Vector.Storable         as V
import           GHC.TypeLits                 hiding (natVal)
import qualified Numeric.LinearAlgebra        as NLA
import qualified Numeric.LinearAlgebra        as D
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra.Static as H
import qualified Numeric.LinearAlgebra.Devel  as U

import           System.Random.MWC

import Debug.Trace

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

data Tensor 
  = T1 Nat Shape 
  | T2 Nat Shape 
  | T3 Nat Shape

-- | Tensors represent batches of shapes
--data Tensor (n :: Nat) (s :: Shape) where
--  T1 :: ( KnownNat batches
--        , KnownNat rows )
--     => batches 
--     -> 'D1 rows 
--     -> Tensor batches ('D1 rows)
--  
--  T2 :: ( KnownNat batches
--        , KnownNat rows
--        , KnownNat columns )
--     => batches 
--     -> 'D2 rows columns 
--     -> Tensor batches ('D2 rows columns)
--  
--  T3 :: ( KnownNat batches
--        , KnownNat rows
--        , KnownNat columns
--        , KnownNat channels )
--     => batches 
--     -> 'D3 rows columns channels
--     -> Tensor batches ('D3 rows columns channels)

data T (n :: Nat) (s :: Shape) where
  -- | Batch of one dimensional vectors. Depth, Row
  T1D :: ( KnownNat batches 
         , KnownNat rows)
      => L batches rows 
      -> T batches ('D1 rows)
  
  -- | Batch of two dimensional vectors. Depth, Row, Column
  T2D :: ( KnownNat batches 
         , KnownNat rows
         , KnownNat columns 
         , KnownNat (rows * columns))
      => L batches (rows * columns) 
      -> T batches ('D2 rows columns)
  
  -- | Batch of three dimensional vectors. Depth, Row, Column, Channels
  T3D :: ( KnownNat batches 
         , KnownNat rows
         , KnownNat columns 
         , KnownNat channels
         , KnownNat (rows * columns * channels))
      => L batches (rows * columns * channels)
      -> T batches ('D3 rows columns channels)

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
#else
data instance Sing (n :: Shape) where
  D1Sing :: Sing a -> Sing ('D1 a)
  D2Sing :: Sing a -> Sing b -> Sing ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> Sing ('D3 a b c)
#endif

instance KnownNat a => SingI ('D1 a) where
  sing = D1Sing sing
instance (KnownNat a, KnownNat b) => SingI ('D2 a b) where
  sing = D2Sing sing sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat (a * c)) => SingI ('D3 a b c) where
  sing = D3Sing sing sing sing

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
          ) :: PutM ()

  get = do
    Just i <- fromStorable . V.fromList <$> getListOf get
    return i

-- Helper function for creating the number instances
n1 :: ( forall a. Floating a => a -> a ) -> S x -> S x
n1 f (S1D x) = S1D (f x)
n1 f (S2D x) = S2D (f x)
n1 f (S3D x) = S3D (f x)

-- Helper function for creating the number instances
n2 :: ( forall a. Floating a => a -> a -> a ) -> S x -> S x -> S x
n2 f (S1D x) (S1D y) = S1D (f x y)
n2 f (S2D x) (S2D y) = S2D (f x y)
n2 f (S3D x) (S3D y) = S3D (f x y)

-- Helper function for creating the number instances
nk :: forall x. SingI x => RealNum -> S x
nk x = case (sing :: Sing x) of
  D1Sing SNat ->
    S1D (konst x)

  D2Sing SNat SNat ->
    S2D (konst x)

  D3Sing SNat SNat SNat ->
    S3D (konst x)

batchMap :: forall x y batches. ( SingI x, SingI y, KnownNat batches ) => (S x -> S y) -> T batches x -> T batches y
batchMap f t 
  = case (sing :: Sing x, sing :: Sing y) of 
    (D1Sing SNat, D1Sing SNat) -> 
      batchMap1D f t
    
    (D2Sing SNat SNat, D2Sing SNat SNat) ->
      batchMap2D f t
    
    (D3Sing SNat SNat SNat, D3Sing SNat SNat SNat) ->
      batchMap3D f t


batchMap1D :: forall a x batches. ( KnownNat batches, KnownNat x ) 
           => (S ('D1 a) -> S ('D1 x)) -> T batches ('D1 a) ->  T batches ('D1 x)
batchMap1D f (T1D m) 
  = let b  = fromIntegral $ natVal (Proxy :: Proxy batches)
        r  = fromIntegral $ natVal (Proxy :: Proxy x)
        m' = H.extract m
    in T1D $ fromJust . H.create $ U.matrixFromVector U.RowMajor b r $ V.concat $ map (\i -> batchMap1D' f $ m' D.! i) [0..b - 1]

batchMap1D' :: KnownNat a => (S ('D1 a) -> S ('D1 x)) -> Vector RealNum -> Vector RealNum
batchMap1D' f v = (\(S1D v) -> H.extract v) $ f (S1D $ fromJust $ H.create v)

batchMap2D :: forall a b x y batches. ( KnownNat batches, KnownNat x, KnownNat y , KnownNat (x * y)) 
           => (S ('D2 a b) -> S ('D2 x y)) -> T batches ('D2 a b) ->  T batches ('D2 x y)
batchMap2D f (T2D m)
  = let b  = fromIntegral $ natVal (Proxy :: Proxy batches)
        r  = fromIntegral $ natVal (Proxy :: Proxy x)
        c  = fromIntegral $ natVal (Proxy :: Proxy y)
        m' = H.extract m
    in T2D $ fromJust . H.create $ U.matrixFromVector U.RowMajor b (r * c) $ V.concat $ map (\i -> batchMap2D' f $ m' D.! i) [0..b - 1]

batchMap2D' :: forall a b x y. (KnownNat a, KnownNat b) => (S ('D2 a b) -> S ('D2 x y)) -> Vector RealNum -> Vector RealNum
batchMap2D' f v = 
  let r      = fromIntegral $ natVal (Proxy :: Proxy a)
      c      = fromIntegral $ natVal (Proxy :: Proxy b)
  in (\(S2D m') -> D.flatten $ H.extract m') $ f (S2D $ fromJust . H.create $ U.matrixFromVector U.RowMajor r c v)

batchMap3D :: forall a b c x y z batches. 
              ( KnownNat batches
              , KnownNat a
              , KnownNat b
              , KnownNat c
              , KnownNat (a * c)
              , KnownNat x
              , KnownNat y
              , KnownNat z
              , KnownNat (x * y * z)) 
           => (S ('D3 a b c) -> S ('D3 x y z)) -> T batches ('D3 a b c) ->  T batches ('D3 x y z)
batchMap3D f (T3D m)
  = let b  = fromIntegral $ natVal (Proxy :: Proxy batches)
        r  = fromIntegral $ natVal (Proxy :: Proxy x)
        c  = fromIntegral $ natVal (Proxy :: Proxy y)
        d  = fromIntegral $ natVal (Proxy :: Proxy z)
        m' = H.extract m
    in T3D $ fromJust . H.create $ U.matrixFromVector U.RowMajor b (r * c * d) $ V.concat $ map (\i -> batchMap3D' f $ m' D.! i) [0..b - 1]

batchMap3D' :: forall a b c x y z. (KnownNat a, KnownNat b, KnownNat c, KnownNat (a * c)) => (S ('D3 a b c) -> S ('D3 x y z)) -> Vector RealNum -> Vector RealNum
batchMap3D' f v = 
  let r = fromIntegral $ natVal (Proxy :: Proxy a)
      c = fromIntegral $ natVal (Proxy :: Proxy b)
      d = fromIntegral $ natVal (Proxy :: Proxy c)
  in (\(S3D m) -> D.flatten $ H.extract m) $ f (S3D $ fromJust $ H.create $ U.matrixFromVector U.RowMajor (r * d) c v)