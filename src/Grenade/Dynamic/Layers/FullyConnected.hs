{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE OverloadedLabels      #-}

module Grenade.Dynamic.Layers.FullyConnected 
  ( SpecFullyConnected (..)
  , specFullyConnected
  , fullyConnected
  ) where

import           Data.Proxy
import           Data.Reflection                     (reifyNat)
import           Data.Singletons
import           Data.Singletons.Prelude.Num         ((%*))
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static        hiding (build, toRows, (&),
                                                      (|||), size)

import           Grenade.Core
import           Grenade.Utils.ListStore
import           Grenade.Dynamic.Internal.Build
import           Grenade.Dynamic.Specification
import           Grenade.Layers.FullyConnected

-------------------- DynamicNetwork instance --------------------

instance (KnownNat i, KnownNat o) => FromDynamicLayer (FullyConnected i o) where
  fromDynamicLayer _ _ _ = SpecNetLayer $ SpecFullyConnected (natVal (Proxy :: Proxy i)) (natVal (Proxy :: Proxy o))

instance ToDynamicLayer SpecFullyConnected where
  toDynamicLayer wInit gen (SpecFullyConnected nrI nrO) =
    reifyNat nrI $ \(pxInp :: (KnownNat i) => Proxy i) ->
      reifyNat nrO $ \(pxOut :: (KnownNat o') => Proxy o') ->
        case singByProxy pxInp %* singByProxy pxOut of
          SNat -> do
            (layer :: FullyConnected i o') <- randomFullyConnected wInit gen
            return $ SpecLayer layer (sing :: Sing ('D1 i)) (sing :: Sing ('D1 o'))

-- | Make a specification of a fully connected layer (see Grenade.Dynamic.Build for a user-interface to specifications).
specFullyConnected :: Integer -> Integer -> SpecNet
specFullyConnected nrI nrO = SpecNetLayer $ SpecFullyConnected nrI nrO


-- | A Fully-connected layer with input dimensions as given in last output layer and output dimensions specified. 1D only!
fullyConnected :: Integer -> BuildM ()
fullyConnected rows = do
  (inRows, _, _) <- buildRequireLastLayerOut Is1D
  buildAddSpec (SpecNetLayer $ SpecFullyConnected inRows rows)
  buildSetLastLayer (rows, 1, 1)


-------------------- GNum instances --------------------

instance (KnownNat i, KnownNat o) => GNum (FullyConnected i o) where
  s |* FullyConnected w store = FullyConnected (s |* w) (s |* store)
  FullyConnected w1 store1 |+ FullyConnected w2 store2 = FullyConnected (w1 |+ w2) (store1 |+ store2)
  gFromRational r = FullyConnected (gFromRational r) mkListStore

instance (KnownNat i, KnownNat o) => GNum (FullyConnected' i o) where
  s |* FullyConnected' b w = FullyConnected' (dvmap (fromRational s *) b) (dmmap (fromRational s *) w)
  FullyConnected' b1 w1 |+ FullyConnected' b2 w2 = FullyConnected' (b1 + b2) (w1 + w2)
  gFromRational r = FullyConnected' (fromRational r) (fromRational r)
