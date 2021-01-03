module Grenade.Dynamic.Recurrent.Layers.LSTM where

import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static

import           Grenade.Dynamic.Specification
import           Grenade.Recurrent.Layers.LSTM

-------------------- GNum instances --------------------

instance (KnownNat i, KnownNat o) => GNum (LSTM i o) where
  n |* (LSTM w m) = LSTM (n |* w) (n |* m)
  (LSTM w1 m1) |+ (LSTM w2 m2) = LSTM (w1 |+ w2) (m1 |+ m2)
  gFromRational r = LSTM (gFromRational r) (LSTMWeights w0 u0 v0 w0 u0 v0 w0 u0 v0 w0 v0)
    where
      v0 = konst 0
      w0 = konst 0
      u0 = konst 0


instance (KnownNat i, KnownNat o) => GNum (LSTMWeights i o) where
  n |* (LSTMWeights wf uf bf wi ui bi wo uo bo wc bc) =
    LSTMWeights
      (fromRational n * wf)
      (fromRational n * uf)
      (fromRational n * bf)
      (fromRational n * wi)
      (fromRational n * ui)
      (fromRational n * bi)
      (fromRational n * wo)
      (fromRational n * uo)
      (fromRational n * bo)
      (fromRational n * wc)
      (fromRational n * bc)
  (LSTMWeights wf1 uf1 bf1 wi1 ui1 bi1 wo1 uo1 bo1 wc1 bc1) |+ (LSTMWeights wf2 uf2 bf2 wi2 ui2 bi2 wo2 uo2 bo2 wc2 bc2) =
    LSTMWeights (wf1 + wf2) (uf1 + uf2) (bf1 + bf2) (wi1 + wi2) (ui1 + ui2) (bi1 + bi2) (wo1 + wo2) (uo1 + uo2) (bo1 + bo2) (wc1 + wc2) (bc1 + bc2)
  gFromRational r = LSTMWeights w0 u0 v0 w0 u0 v0 w0 u0 v0 w0 v0
    where
      v0 = fromRational r
      w0 = fromRational r
      u0 = fromRational r
