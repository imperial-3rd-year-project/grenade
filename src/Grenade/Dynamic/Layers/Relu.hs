{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Relu
  ( SpecRelu (..)
  , specRelu1D
  , specRelu2D
  , specRelu3D
  , relu
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
import           Grenade.Layers.Relu

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Relu where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecRelu (tripleFromSomeShape inp)

instance ToDynamicLayer SpecRelu where
  toDynamicLayer _ _ (SpecRelu (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Relu (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Relu (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Relu (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specRelu1D :: Integer -> SpecNet
specRelu1D i = specRelu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specRelu2D :: (Integer, Integer) -> SpecNet
specRelu2D (i, j) = specRelu3D (i, j, 1)

-- | Create a specification for a elu layer.
specRelu3D :: (Integer, Integer, Integer) -> SpecNet
specRelu3D = SpecNetLayer . SpecRelu


-- | Add a Relu layer to your build.
relu :: BuildM ()
relu = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecRelu


-------------------- GNum instances --------------------

instance GNum Relu where
  _ |* Relu = Relu
  _ |+ Relu = Relu
  gFromRational _ = Relu
