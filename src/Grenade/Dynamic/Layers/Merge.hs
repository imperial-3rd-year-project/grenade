module Grenade.Dynamic.Layers.Merge where

import           Grenade.Dynamic.Specification
import           Grenade.Layers.Merge

-------------------- GNum instances --------------------

instance (GNum x, GNum y) => GNum (Merge x y) where
  n |* (Merge x y) = Merge (n |* x) (n |* y)
  (Merge x y) |+ (Merge x2 y2) = Merge (x |+ x2) (y |+ y2)
  gFromRational r = Merge (gFromRational r) (gFromRational r)
