{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE CPP                  #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE StandaloneDeriving   #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeOperators        #-}
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
  , visualise2D
  , splitChannels
  , combineChannels
  ) where

#if MIN_VERSION_singletons(2,6,0)
import           Data.Kind                    (Type)
#endif

import           Control.DeepSeq              (NFData (..))
import           Data.List.Split              (chunksOf)
import           Data.Maybe                   (fromJust)
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.Vector.Storable         (Vector)
import qualified Data.Vector.Storable         as V
import           GHC.TypeLits                 hiding (natVal)
import qualified Numeric.LinearAlgebra        as NLA
import qualified Numeric.LinearAlgebra.Data   as NLAD
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
  | D4 Nat Nat Nat Nat 
  -- ^ Four dimensional matrix. Depth, Channels, Row, Column

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
         , KnownNat channels
         , KnownNat (rows * channels) )
      => L (rows * channels) columns
      -> S ('D3 rows columns channels)
  
  S4D :: ( KnownNat rows
         , KnownNat columns
         , KnownNat channels
         , KnownNat depth)
      => L (depth * channels * rows) columns
      -> S ('D4 depth channels rows columns)

deriving instance Show (S n)

-- Singleton instances.
--
-- These could probably be derived with template haskell, but this seems
-- clear and makes adding the KnownNat constraints simple.
-- We can also keep our code TH free, which is great.
#if MIN_VERSION_singletons(2,6,0)
-- In singletons 2.6 Sing switched from a data family to a type family.
type instance Sing = SShape

-- | TODO someone 
data SShape :: Shape -> Type where
  D1Sing :: Sing a -> SShape ('D1 a)
  D2Sing :: Sing a -> Sing b -> SShape ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> SShape ('D3 a b c)
  D4Sing :: KnownNat (a * c * d) => Sing a -> Sing b -> Sing c -> Sing d -> SShape ('D4 a b c d)
#else
data instance Sing (n :: Shape) where
  D1Sing :: Sing a -> Sing ('D1 a)
  D2Sing :: Sing a -> Sing b -> Sing ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> Sing ('D3 a b c)
  D4Sing :: KnownNat (a * c * d) => Sing a -> Sing b -> Sing c -> Sing d -> Sing ('D4 a b c d)
#endif

instance KnownNat a => SingI ('D1 a) where
  sing = D1Sing sing
instance (KnownNat a, KnownNat b) => SingI ('D2 a b) where
  sing = D2Sing sing sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat (a * c)) => SingI ('D3 a b c) where
  sing = D3Sing sing sing sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat d) => SingI ('D4 a b c d) where
  sing = D4Sing sing sing sing sing

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
  rnf (S4D x) = rnf x

-- | Generate random data of the desired shape
randomOfShape :: forall x . (SingI x) => IO (S x)
randomOfShape = do
  seed :: Int <- withSystemRandom . asGenST $ \gen -> uniform gen
  return $ case (sing :: Sing x) of
    D1Sing SNat ->
        S1D (H.randomVector seed H.Uniform * 2 - 1)

    D2Sing SNat SNat ->
        S2D (H.uniformSample seed (-1) 1)

    D3Sing SNat SNat SNat ->
        S3D (H.uniformSample seed (-1) 1)
    
    D4Sing SNat SNat SNat SNat ->
        S4D (H.uniformSample seed (-1) 1)

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

    D4Sing SNat SNat SNat SNat ->
      S4D <$> mkL xs
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
            (S4D x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
          ) :: PutM ()

  get = do
    Just i <- fromStorable . V.fromList <$> getListOf get
    return i

-- Helper function for creating the number instances
n1 :: ( forall a. Floating a => a -> a ) -> S x -> S x
n1 f (S1D x) = S1D (f x)
n1 f (S2D x) = S2D (f x)
n1 f (S3D x) = S3D (f x)
n1 f (S4D x) = S4D (f x)

-- Helper function for creating the number instances
n2 :: ( forall a. Floating a => a -> a -> a ) -> S x -> S x -> S x
n2 f (S1D x) (S1D y) = S1D (f x y)
n2 f (S2D x) (S2D y) = S2D (f x y)
n2 f (S3D x) (S3D y) = S3D (f x y)
n2 f (S4D x) (S4D y) = S4D (f x y)

-- | Helper function for creating the number instances
nk :: forall x. SingI x => RealNum -> S x
nk x = case (sing :: Sing x) of
  D1Sing SNat ->
    S1D (H.konst x)

  D2Sing SNat SNat ->
    S2D (H.konst x)

  D3Sing SNat SNat SNat ->
    S3D (H.konst x)
  
  D4Sing SNat SNat SNat SNat ->
    S4D (H.konst x)

-- | TODO Jason
visualise2D :: S ('D2 a b) -> RealNum -> String
visualise2D (S2D mm) max = 
  let m  = H.extract mm
      ms = NLAD.toLists m
      render n' | n' <= 0.2 * max  = ' '
                | n' <= 0.4 * max  = '.'
                | n' <= 0.6 * max  = '-'
                | n' <= 0.8 * max  = '='
                | otherwise =  '#'
      px = (fmap . fmap) render ms
  in unlines px

-- | TODO Theo
splitChannels :: forall rows columns channels.
                 (KnownNat rows, KnownNat columns)
                 => S ('D3 rows columns channels) -> [S ('D2 rows columns)]
splitChannels (S3D x)
  = let r   = fromIntegral $ natVal (Proxy :: Proxy rows)
        rs  = NLA.toRows $ H.extract x
        rs' = chunksOf r rs
        ms  = map (S2D . fromJust . H.create . NLA.fromRows) rs' :: [S ('D2 rows columns)]
    in ms

-- | TODO Theo
combineChannels :: forall rows columns channels.
                 (KnownNat rows, KnownNat columns, KnownNat channels)
                 => [S ('D2 rows columns)] -> S ('D3 rows columns channels)
combineChannels xs
  = let xs' = map (\(S2D x) -> NLA.toRows $ H.extract x) xs :: [[Vector RealNum]]
        xs'' = concat xs'
    in  S3D $ fromJust . H.create . NLA.fromRows $ xs''
