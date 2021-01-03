module Grenade.Dynamic.Layers.Concat where

import           Grenade.Dynamic.Specification
import           Grenade.Layers.Concat

-------------------- DynamicNetwork instance --------------------

-- TODO

-- instance (FromDynamicLayer x, FromDynamicLayer y) => FromDynamicLayer (Concat m x n y) where
--   fromDynamicLayer inp (Concat x y) = SpecNetLayer $ SpecConcat (tripleFromSomeShape inp) (fromDynamicLayer x) (fromDynamicLayer y)


-- instance ToDynamicLayer SpecConcat where
--   toDynamicLayer wInit gen (SpecConcat xSpec ySpec) = do
--     x <- toDynamicLayer wInit gen xSpec
--     y <- toDynamicLayer wInit gen ySpec
--     return $ SpecLayer $ Concat x y


--   toDynamicLayer wInit gen (SpecFullyConnected nrI nrO) =
--     reifyNat nrI $ \(pxInp :: (KnownNat i') => Proxy i') ->
--       reifyNat nrO $ \(pxOut :: (KnownNat o') => Proxy o') ->
--         case (singByProxy pxInp %* singByProxy pxOut, unsafeCoerce (Dict :: Dict ()) :: Dict (i' ~ i), unsafeCoerce (Dict :: Dict ()) :: Dict (o' ~ o)) of
--           (SNat, Dict, Dict) -> do
--             (layer  :: FullyConnected i' o') <- randomFullyConnected wInit gen
--             return $ SpecLayer layer (SomeSing (sing :: Sing ('D1 i'))) (SomeSing (sing :: Sing ('D1 o')))


-- specFullyConnected :: Integer -> Integer -> SpecNet
-- specFullyConnected nrI nrO = SpecNetLayer $ SpecFullyConnected nrI nrO


-------------------- GNum instance --------------------


instance (GNum x, GNum y) => GNum (Concat m x n y) where
  n |* (Concat x y) = Concat (n |* x) (n |* y)
  (Concat x1 y1) |+ (Concat x2 y2) = Concat (x1 |+ x2) (y1 |+ y2)
  gFromRational r = Concat (gFromRational r) (gFromRational r)