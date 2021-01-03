module Grenade.Dynamic.Layers.Crop where

import           Grenade.Dynamic.Specification
import           Grenade.Layers.Crop

-------------------- GNum instances --------------------

instance GNum (Crop l t r b) where
  _ |* x = x
  _ |+ x = x
  gFromRational _ = Crop
