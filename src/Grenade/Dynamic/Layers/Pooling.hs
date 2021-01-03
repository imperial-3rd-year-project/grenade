module Grenade.Dynamic.Layers.Pooling where

import           Grenade.Dynamic.Specification
import           Grenade.Layers.Pooling

-------------------- GNum instance --------------------

instance GNum (Pooling k k' s s') where
  _ |* _ = Pooling
  _ |+ _ = Pooling
  gFromRational _ = Pooling
