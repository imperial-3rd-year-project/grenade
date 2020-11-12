module Grenade.Core.NetworkSettings
    ( NetworkSettings(..)

    ) where


data NetworkSettings = NetworkSettings
  { setDropoutActive :: Bool
  , trainingActive   :: Bool
  }
