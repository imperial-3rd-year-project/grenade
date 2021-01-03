module Grenade.Dynamic.Layers.Pad where

import           Grenade.Dynamic.Specification
import           Grenade.Layers.Pad

-------------------- GNum instance --------------------

instance GNum (Pad l t r b) where
  _ |* _ = Pad
  _ |+ _ = Pad
  gFromRational _ = Pad