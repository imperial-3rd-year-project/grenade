{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Trivial 
  ( SpecTrivial(..)
  , specTrivial1D
  , specTrivial2D
  , specTrivial3D
  , trivial
  ) where

import           Data.Constraint                     (Dict (..))
import           Data.Proxy
import           Data.Reflection                     (reifyNat)
import           Data.Singletons
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits

import           Unsafe.Coerce                       (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic.Network
import           Grenade.Dynamic.Specification
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Trivial

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Trivial where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecTrivial (tripleFromSomeShape inp)

instance ToDynamicLayer SpecTrivial where
  toDynamicLayer _ _ (SpecTrivial (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Trivial (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Trivial (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Trivial (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))

-- | Create a specification for a elu layer.
specTrivial1D :: Integer -> SpecNet
specTrivial1D i = specTrivial3D (i, 1, 1)

-- | Create a specification for a elu layer.
specTrivial2D :: (Integer, Integer) -> SpecNet
specTrivial2D (i, j) = specTrivial3D (i, j, 1)

-- | Create a specification for a elu layer.
specTrivial3D :: (Integer, Integer, Integer) -> SpecNet
specTrivial3D = SpecNetLayer . SpecTrivial

-- | Add a Trivial layer to your build.
trivial :: BuildM ()
trivial = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecTrivial


-------------------- GNum instances --------------------

instance GNum Trivial where
  _ |* Trivial = Trivial
  _ |+ Trivial  = Trivial
  gFromRational _ = Trivial
