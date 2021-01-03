{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Sinusoid
  ( SpecSinusoid (..)
  , specSinusoid1D
  , specSinusoid2D
  , specSinusoid3D
  , sinusoid
  ) where

import           Data.Constraint                (Dict (..))
import           Data.Proxy
import           Data.Reflection                (reifyNat)
import           Data.Singletons
import           Data.Singletons.TypeLits       hiding (natVal)
import           GHC.TypeLits
import           Unsafe.Coerce                  (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic.Internal.Build
import           Grenade.Dynamic.Network
import           Grenade.Dynamic.Specification
import           Grenade.Layers.Sinusoid

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Sinusoid where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecSinusoid (tripleFromSomeShape inp)

instance ToDynamicLayer SpecSinusoid where
  toDynamicLayer _ _ (SpecSinusoid (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Sinusoid (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Sinusoid (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Sinusoid (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specSinusoid1D :: Integer -> SpecNet
specSinusoid1D i = specSinusoid3D (i, 1, 1)

-- | Create a specification for a elu layer.
specSinusoid2D :: (Integer, Integer) -> SpecNet
specSinusoid2D (i,j) = specSinusoid3D (i,j,1)

-- | Create a specification for a elu layer.
specSinusoid3D :: (Integer, Integer, Integer) -> SpecNet
specSinusoid3D = SpecNetLayer . SpecSinusoid


-- | Add a Sinusoid layer to your build.
sinusoid :: BuildM ()
sinusoid = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecSinusoid


-------------------- GNum instances --------------------

instance GNum Sinusoid where
  _ |* Sinusoid = Sinusoid
  _ |+ Sinusoid = Sinusoid
  gFromRational _ = Sinusoid
