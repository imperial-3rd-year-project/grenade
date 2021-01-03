{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Elu 
  ( SpecElu (..)
  , specElu1D
  , specElu2D
  , specElu3D
  , elu
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
import           Grenade.Dynamic.Specification
import           Grenade.Dynamic.Network
import           Grenade.Layers.Elu

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Elu where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecElu (tripleFromSomeShape inp)

instance ToDynamicLayer SpecElu where
  toDynamicLayer _ _ (SpecElu (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Elu (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Elu (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Elu (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specElu1D :: Integer -> SpecNet
specElu1D i = specElu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specElu2D :: (Integer, Integer) -> SpecNet
specElu2D (i,j) = specElu3D (i,j,1)

-- | Create a specification for a elu layer.
specElu3D :: (Integer, Integer, Integer) -> SpecNet
specElu3D = SpecNetLayer . SpecElu

-- | Add a Elu layer to your build.
elu :: BuildM ()
elu = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecElu


-------------------- GNum instances --------------------


instance GNum Elu where
  _ |* Elu = Elu
  _ |+ Elu = Elu
  gFromRational _ = Elu

