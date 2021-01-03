{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Logit 
  ( SpecLogit (..)
  , specLogit1D
  , specLogit2D
  , specLogit3D
  , logit
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
import           Grenade.Layers.Logit

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Logit where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecLogit (tripleFromSomeShape inp)

instance ToDynamicLayer SpecLogit where
  toDynamicLayer _ _ (SpecLogit (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Logit (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Logit (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Logit (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specLogit1D :: Integer -> SpecNet
specLogit1D i = specLogit3D (i, 1, 1)

-- | Create a specification for a elu layer.
specLogit2D :: (Integer, Integer) -> SpecNet
specLogit2D (i, j) = specLogit3D (i, j, 1)

-- | Create a specification for a elu layer.
specLogit3D :: (Integer, Integer, Integer) -> SpecNet
specLogit3D = SpecNetLayer . SpecLogit


-- | Add a Logit layer to your build.
logit :: BuildM ()
logit = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecLogit


-------------------- GNum instances --------------------

instance GNum Logit where
  _ |* x = x
  _ |+ x = x
  gFromRational _ = Logit
