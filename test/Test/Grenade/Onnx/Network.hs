{-# LANGUAGE CPP                  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE PolyKinds            #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TemplateHaskell      #-}
{-# LANGUAGE TupleSections        #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE ViewPatterns         #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeApplications     #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

module Test.Grenade.Onnx.Network where

import Test.Grenade.Onnx.Utils

import Grenade

import           Data.Either.Combinators
import qualified Data.Text              as T
import           Data.List                 (mapAccumL)
import           Data.Proxy

import           Data.Singletons.Prelude.List
import           Data.Kind                 (Type)
import           GHC.TypeLits

import           Hedgehog
import           Hedgehog.Gen           as Gen
import           Hedgehog.Range         as Range

import           Data.ProtoLens            (defMessage)
import           Data.ProtoLens.Encoding   (encodeMessage, decodeMessage)
import           Data.ProtoLens.Labels     ()
import           Proto.Onnx             as P

import           Lens.Micro

import Test.Grenade.Onnx.FakeLayers

type Activation = ActFakeLayer "activation"

type family ConsLayer (layer :: Type) (shape :: Shape) (network :: Type) :: Type where
  ConsLayer x h (Network xs hs) = Network (x ': xs) (h ': hs)

type family BuildNetwork (layers :: [Type]) (shape :: Shape) :: Type where
  BuildNetwork '[]       h = Network '[] '[h]
  BuildNetwork (x ': xs) h = ConsLayer x h (BuildNetwork xs h)
  
type FiveActivationLayers = Replicate 5 Activation
type FiveActivationsNetwork = BuildNetwork FiveActivationLayers ('D1 1)

type FiveActivationsFailNetwork =
  BuildNetwork (FiveActivationLayers ++ '[AlwaysFail "always_fail"]) ('D1 1)

mkModel :: [P.NodeProto] -> P.ModelProto
mkModel = (defMessage &) . (#graph . #node .~)

class OnnxStorable a where
  toOnnxNode :: Proxy a -> T.Text -> T.Text -> P.NodeProto
  toOnnxNode = undefined
  toOnnx :: Proxy a -> T.Text -> T.Text -> [T.Text] -> ([T.Text], [P.NodeProto])
  toOnnx proxy inName outName names = (names, [toOnnxNode proxy inName outName])
  toOnnxRand :: Proxy a -> T.Text -> T.Text -> [T.Text] -> Gen ([T.Text], [P.NodeProto])
  toOnnxRand proxy inName outName names = return (toOnnx proxy inName outName names)
  {-# MINIMAL toOnnxNode | toOnnx | toOnnxRand #-}

defaultNode :: OnnxOperator a => Proxy a -> T.Text -> T.Text -> P.NodeProto
defaultNode proxy inName outName = defMessage & #opType .~ head (onnxOpTypeNames proxy)
                                              & #input  .~ [inName]
                                              & #output .~ [outName]

instance KnownSymbol layer => OnnxStorable (FakeParLayer layer Trivial Trivial) where
  toOnnxNode = defaultNode

instance OnnxOperator a => OnnxStorable (Lift (LoadActivation a)) where
  toOnnxNode = defaultNode

instance (OnnxLoadableParallel a x y, OnnxStorable x, OnnxStorable y)
      => OnnxStorable (Lift (LoadParallel a)) where
  toOnnxRand proxy inName outName (mergeName : names) = do
    (names', xNodes)  <- toOnnxRand (Proxy :: Proxy x) inName mergeName names
    (names'', yNodes) <- toOnnxRand (Proxy :: Proxy y) inName mergeName names'
    let parNode = defaultNode proxy mergeName outName
    parNodes <- randomMerge [xNodes, yNodes]
    return (names'', parNodes ++ [parNode])

instance OnnxStorable x => OnnxStorable (Network '[x] '[i, h]) where
  toOnnxRand _ = toOnnxRand (Proxy :: Proxy x)

instance (OnnxStorable x, OnnxStorable x', OnnxStorable (Network (x' ': xs) (h ': hs)))
       => OnnxStorable (Network (x ': x' ': xs) (i ': h ': hs)) where
  toOnnxRand _ inName outName (midName : names) = do
    (names', nodes) <- toOnnxRand (Proxy :: Proxy x) inName midName names
    ((nodes ++) <$>) <$> toOnnxRand (Proxy :: Proxy (Network (x' ': xs) (h ': hs))) midName outName names'

canLoadFrom :: forall a b m. (OnnxLoadable a, OnnxStorable b, Monad m)
            => Proxy a -> Proxy b -> PropertyT m ()
canLoadFrom _ proxy = do
  model <- forAll $ mkModel <$> genNodes proxy
  evalEither (loadModel model :: Either OnnxLoadFailure a)
  success

canLoad :: forall a m. (OnnxStorable a, OnnxLoadable a, Monad m)
        => Proxy a -> PropertyT m ()
canLoad proxy = proxy `canLoadFrom` proxy

loadFailsFromAt :: forall a b m. (Show a, OnnxLoadable a, OnnxStorable b, Monad m)
                => Proxy a -> Proxy b -> T.Text -> PropertyT m ()
loadFailsFromAt _ proxy opType = do
  model <- forAll $ mkModel <$> genNodes proxy

  let loaded :: Either OnnxLoadFailure a
      loaded = loadModel model
  failureInfo <- evalEither (swapEither loaded) >>= evalNF

  failedAt <- evalMaybe' (failureInfo ^. currentNode)
  failedAt' :: P.NodeProto <- evalEither (castMessage failedAt)

  failedAt' ^. #opType === opType

loadFailsAt :: forall a m. (Show a, OnnxLoadable a, OnnxStorable a, Monad m)
            => Proxy a -> T.Text -> PropertyT m ()
loadFailsAt proxy = loadFailsFromAt proxy proxy

genNodes :: OnnxStorable x => Proxy x -> Gen [P.NodeProto]
genNodes proxy = snd <$> toOnnxRand proxy inName outName edges
  where (inName : outName : edges) = edgeNames

prop_can_load_linear_acts =
  withTests 1 . property $ canLoad (Proxy @FiveActivationsNetwork)

prop_fail_in_layer_fails_network = withTests 1 . property $
  (Proxy @FiveActivationsFailNetwork) `loadFailsAt` "always_fail"

prop_not_consume_all_nodes_fails_network = withTests 1 . property $
  loadFailsFromAt (Proxy @FiveActivationsNetwork)
                  (Proxy @(BuildNetwork (Activation ': FiveActivationLayers) ('D1 1)))
                  "activation"

type Bypass = BypassFakeLayer "bypass"
type ThreeBypassFiveActivationsLayer = Replicate 3 Bypass ++ FiveActivationLayers
type ThreeBypassFiveActivationsNetwork = BuildNetwork ThreeBypassFiveActivationsLayer ('D1 1)

prop_bypass_layers_bypassed = withTests 1 . property $
  (Proxy @ThreeBypassFiveActivationsNetwork) `canLoadFrom` (Proxy @FiveActivationsNetwork)

prop_mismatch_opType_fails_load = withTests 1 . property $
  loadFailsFromAt (Proxy @(ActFakeLayer "Not act"))
                  (Proxy @(BuildNetwork '[Activation] ('D1 1)))
                  "activation"

type ParLayer = BuildNetwork '[Activation, ParFakeLayer "par" Activation Activation] ('D1 1)

prop_can_load_par = withTests 1 . property $ canLoad (Proxy @ParLayer)

type ParLeftBranchFailNetwork =
  BuildNetwork '[ Activation
                , ParFakeLayer "par" FiveActivationsFailNetwork FiveActivationsNetwork
                ] ('D1 1)
prop_par_branch_fail_fails_par = withTests 1 . property $
  loadFailsAt (Proxy @ParLeftBranchFailNetwork) "par"

tests :: IO Bool
tests = checkParallel $$(discover)
