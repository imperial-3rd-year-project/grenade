{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.Dynamic.Layers.Dropout 
  ( SpecDropout (..)
  , specDropout
  , dropout
  , dropoutWithSeed
  ) where

import           Data.Proxy
import           Data.Reflection                     (reifyNat)
import           Data.Singletons
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Types
import           Grenade.Dynamic.Internal.Build
import           Grenade.Dynamic.Specification
import           Grenade.Dynamic.Network
import           Grenade.Layers.Dropout

-------------------- DynamicNetwork instance --------------------

instance (KnownNat pct) => FromDynamicLayer (Dropout pct) where
  fromDynamicLayer inp _ (Dropout _ seed) = case tripleFromSomeShape inp of
    (rows, 1, 1) -> SpecNetLayer $ SpecDropout rows rate (Just seed)
    _            -> error "Dropout is only allows for vectors, i.e. 1D spaces."
    where rate = (/100) $ fromIntegral $ max 0 $ min 100 $ natVal (Proxy :: Proxy pct)

instance ToDynamicLayer SpecDropout where
  toDynamicLayer _ gen (SpecDropout rows rate mSeed) =
    reifyNat rows $ \(_ :: (KnownNat i) => Proxy i) ->
    reifyNat (round $ 100 * rate) $ \(_ :: (KnownNat pct) => Proxy pct) ->
    case mSeed of
      Just seed -> return $ SpecLayer (Dropout True seed :: Dropout pct) (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))
      Nothing -> do
        layer <-  randomDropout gen
        return $ SpecLayer (layer :: Dropout pct) (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))


-- | Create a specification for a droput layer by providing the input size of the vector (1D allowed only!), a rate of nodes to keep (e.g. 0.95) and maybe a seed.
specDropout :: Integer -> RealNum -> Maybe Int -> SpecNet
specDropout i rate seed = SpecNetLayer $ SpecDropout i rate seed

-- | Create a dropout layer with the specified keep rate of nodes. The seed will be randomly initialized when the network is created. See also @dropoutWithSeed@.
dropout :: RealNum -> BuildM ()
dropout ratio = dropoutWithSeed ratio Nothing

-- | Create a dropout layer with the specified keep rate of nodes. The seed will be randomly initialized when the network is created. See also @dropoutWithSeed@.
dropoutWithSeed :: RealNum -> Maybe Int -> BuildM ()
dropoutWithSeed ratio mSeed = buildRequireLastLayerOut Is1D >>= \(i, _, _) -> buildAddSpec (specDropout i ratio mSeed)


-------------------- GNum instance --------------------

instance GNum (Dropout pct) where
  _ |* x = x
  _ |+ x = x
  gFromRational r = Dropout True (round r)
