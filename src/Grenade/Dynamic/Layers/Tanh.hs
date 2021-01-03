{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Tanh 
  ( SpecTanh (..)
  , specTanh1D
  , specTanh2D
  , specTanh3D
  , specTanh
  , tanhLayer
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
import           Grenade.Layers.Tanh

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Tanh where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecTanh (tripleFromSomeShape inp)

instance ToDynamicLayer SpecTanh where
  toDynamicLayer _ _ (SpecTanh (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Tanh (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Tanh (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Tanh (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a Tanh layer.
specTanh1D :: Integer -> SpecNet
specTanh1D i = specTanh3D (i, 1, 1)

-- | Create a specification for a Tanh layer.
specTanh2D :: (Integer, Integer) -> SpecNet
specTanh2D (i, j) = specTanh3D (i, j, 1)

-- | Create a specification for a Tanh layer.
specTanh3D :: (Integer, Integer, Integer) -> SpecNet
specTanh3D = SpecNetLayer . SpecTanh

-- | Create a specification for a Tanh layer.
specTanh :: (Integer, Integer, Integer) -> SpecNet
specTanh = SpecNetLayer . SpecTanh

-- | Add a Tanh layer to your build.
tanhLayer :: BuildM ()
tanhLayer = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecTanh


-------------------- GNum instances --------------------

instance GNum Tanh where
  _ |* Tanh = Tanh
  _ |+ Tanh = Tanh
  gFromRational _ = Tanh