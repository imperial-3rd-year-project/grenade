{-# LANGUAGE CPP                       #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE FlexibleInstances         #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE KindSignatures            #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeOperators             #-}
{-# LANGUAGE UndecidableInstances      #-}
module Grenade.Dynamic.Recurrent.Layers.BasicRecurrent where

import           GHC.TypeLits

import           Grenade.Dynamic.Specification
import           Grenade.Recurrent.Layers.BasicRecurrent

-------------------- GNum instances --------------------

instance (KnownNat i, KnownNat o, KnownNat (i + o)) => GNum (BasicRecurrent i o) where
  n |* (BasicRecurrent wB mB mA nM) = BasicRecurrent (fromRational n * wB) (fromRational n * mB) (fromRational n * mA) (fromRational n * nM)
  (BasicRecurrent wB mB mA nM) |+ (BasicRecurrent wB2 mB2 a2 nM2) = BasicRecurrent (wB + wB2) (mB + mB2) (mA + a2) (nM + nM2)
  gFromRational r = BasicRecurrent (fromRational r) 0 (fromRational r) 0
