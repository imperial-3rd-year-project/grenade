{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}

module Grenade.Dynamic.Layers.Softmax 
  ( SpecSoftmax (..)
  , specSoftmax
  , softmaxLayer
  ) where

import           Data.Proxy
import           Data.Reflection                     (reifyNat)
import           Data.Singletons
import           Data.Singletons.TypeLits            hiding (natVal)

import           Grenade.Core
import           Grenade.Dynamic.Network
import           Grenade.Dynamic.Specification
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Softmax

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Softmax where
  fromDynamicLayer inp _ Softmax = case tripleFromSomeShape inp of
    (rows, 1, 1) -> SpecNetLayer $ SpecSoftmax rows
    _ -> error "Error in specification: The layer Softmax may only be used with 1D input!"

instance ToDynamicLayer SpecSoftmax where
  toDynamicLayer _ _ (SpecSoftmax rows) =
    reifyNat rows $ \(_ :: (KnownNat i) => Proxy i) ->
    return $ SpecLayer Softmax (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))


-- | Create a specification for a elu layer.
specSoftmax :: Integer -> SpecNet
specSoftmax = SpecNetLayer . SpecSoftmax


-- | Add a Softmax layer to your build.
softmaxLayer :: BuildM ()
softmaxLayer = buildRequireLastLayerOut Is1D >>= buildAddSpec . SpecNetLayer . SpecSoftmax . fst3
  where
    fst3 (x, _, _) = x

-------------------- GNum instances --------------------


instance GNum Softmax where
  _ |* Softmax = Softmax
  _ |+ Softmax = Softmax
  gFromRational _ = Softmax
