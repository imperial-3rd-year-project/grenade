{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Gelu 
  ( SpecGelu (..)
  , specGelu1D
  , specGelu2D
  , specGelu3D
  , gelu
  ) where

import           Data.Constraint                     (Dict (..))
import           Data.Proxy
import           Data.Reflection                     (reifyNat)
import           Data.Singletons
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits

import           Unsafe.Coerce                       (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic.Internal.Build
import           Grenade.Dynamic.Network
import           Grenade.Dynamic.Specification
import           Grenade.Layers.Gelu

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Gelu where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecGelu (tripleFromSomeShape inp)

instance ToDynamicLayer SpecGelu where
  toDynamicLayer _ _ (SpecGelu (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Gelu (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Gelu (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Gelu (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specGelu1D :: Integer -> SpecNet
specGelu1D i = specGelu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specGelu2D :: (Integer, Integer) -> SpecNet
specGelu2D (i, j) = specGelu3D (i, j, 1)

-- | Create a specification for a elu layer.
specGelu3D :: (Integer, Integer, Integer) -> SpecNet
specGelu3D = SpecNetLayer . SpecGelu


-- | Add a Gelu layer to your build.
gelu :: BuildM ()
gelu = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecGelu


-------------------- GNum instances --------------------

instance GNum Gelu where
  _ |* Gelu = Gelu
  _ |+ Gelu = Gelu
  gFromRational _ = Gelu