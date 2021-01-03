{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.LeakyRelu
  ( SpecLeakyRelu (..)
  , specLeakyRelu1D
  , specLeakyRelu2D
  , specLeakyRelu3D
  , leakyRelu
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
import           Grenade.Layers.LeakyRelu
import           Grenade.Types

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer LeakyRelu where
  fromDynamicLayer inp _ (LeakyRelu alpha) = SpecNetLayer $ SpecLeakyRelu (tripleFromSomeShape inp) alpha

instance ToDynamicLayer SpecLeakyRelu where
  toDynamicLayer _ _ (SpecLeakyRelu (rows, cols, depth) alpha) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer (LeakyRelu alpha) (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer (LeakyRelu alpha) (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer (LeakyRelu alpha) (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specLeakyRelu1D :: Integer -> RealNum -> SpecNet
specLeakyRelu1D i = specLeakyRelu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specLeakyRelu2D :: (Integer, Integer) -> RealNum -> SpecNet
specLeakyRelu2D (i, j) = specLeakyRelu3D (i, j, 1)

-- | Create a specification for a elu layer.
specLeakyRelu3D :: (Integer, Integer, Integer) -> RealNum -> SpecNet
specLeakyRelu3D d = SpecNetLayer . SpecLeakyRelu d


-- | Add a LeakyRelu layer to your build.
leakyRelu :: RealNum -> BuildM ()
leakyRelu alpha = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . (\x -> SpecLeakyRelu x alpha)


-------------------- GNum instances --------------------

instance GNum LeakyRelu where
  _ |* x = x
  _ |+ x = x
  gFromRational alpha = LeakyRelu (fromRational alpha)
