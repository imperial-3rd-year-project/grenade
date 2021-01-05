{-|
Module      : Grenade.Core.NetworkSettings
Description : Defines datatype representing NN settings
-}

module Grenade.Core.NetworkSettings
    ( NetworkSettings(..)

    ) where


data NetworkSettings = NetworkSettings
  { setDropoutActive :: Bool
  , trainingActive   :: Bool
  }
