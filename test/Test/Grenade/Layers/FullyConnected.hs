{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.FullyConnected where

import           Data.Constraint               (Dict (..))
import           Data.Proxy
import           Data.Singletons               ()
import           GHC.TypeLits
import           Unsafe.Coerce                 (unsafeCoerce)
import           Data.Kind                     (Type)
import           Hedgehog
import qualified Hedgehog.Gen                as Gen
import qualified Hedgehog.Range              as Range
import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix
import           Test.Utils.Rnf
import           Test.Grenade.Layers.Internal.Reference
import           Grenade.Core.Optimizer
import           System.Random.MWC             (create)
import qualified Numeric.LinearAlgebra.Data   as D
import           Numeric.LinearAlgebra.Static (R)
import qualified Numeric.LinearAlgebra.Static as H
import           Data.Serialize
import           Data.Either

import           Grenade.Core
import           Grenade.Layers.FullyConnected
import           Grenade.Utils.ListStore


data OpaqueFullyConnected :: Type where
     OpaqueFullyConnected :: (KnownNat i, KnownNat o, KnownNat (i * o)) => FullyConnected i o -> OpaqueFullyConnected

instance Show OpaqueFullyConnected where
    show (OpaqueFullyConnected n) = show n

genOpaqueFullyConnected :: Gen OpaqueFullyConnected
genOpaqueFullyConnected = do
  input   :: Integer  <- choose 2 100
  output  :: Integer  <- choose 1 100
  let Just input'      = someNatVal input
  let Just output'     = someNatVal output
  case (input', output') of
    (SomeNat (Proxy :: Proxy i'), SomeNat (Proxy :: Proxy o')) ->
      case (unsafeCoerce (Dict :: Dict ()) :: Dict (KnownNat (i' * o'))) of
        Dict -> do
          wB    <- randomVector
          bM    <- randomVector
          wN    <- uniformSample
          kM    <- uniformSample
          return . OpaqueFullyConnected $ (FullyConnected (FullyConnected' wB wN) (ListStore 0 [Just $ FullyConnected' bM kM]) :: FullyConnected i' o')

prop_fully_connected_run_forwards_backwards :: Property
prop_fully_connected_run_forwards_backwards = property $ do
  OpaqueFullyConnected (fc :: FullyConnected i o) <- blindForAll genOpaqueFullyConnected
  let (FullyConnected (FullyConnected' b w) _) = fc
  input <- forAll randomVector
  let (ntape, nout)  = naiveFullyConnectedRunForwards w b input
      (nnw, nnb, nd) = naiveFullyConnectedBackprop w (ntape, nout)
      (tape, out :: S ('D1 o)) = runForwards fc (S1D input)
      (grad, d   :: S ('D1 i)) = runBackwards fc tape out
      FullyConnected' nb nw = grad
  assert $ allClose out        (S1D nout)
  assert $ allClose d          (S1D nd)
  assert $ allClose (S2D nw)   (S2D nnw)
  assert $ allClose (S1D nb)   (S1D nnb)
  assert $ allClose (S1D tape) (S1D ntape)

prop_fully_connected_rnf :: Property
prop_fully_connected_rnf = property $ do
  gen <- evalIO create
  fc :: FullyConnected 3 5 <- evalIO $ createRandomWith UniformInit gen
  let (FullyConnected fc'@(FullyConnected' b w) _) = fc

  -- Evaluates the weights and biases
  err0 <- evalIO $ tryEvalRnf (FullyConnected (error expectedErrStr) mkListStore)
  assert $ rnfRaisedErr err0

  -- Evaluates the list store
  err1 <- evalIO $ tryEvalRnf (FullyConnected fc' (error expectedErrStr))
  assert $ rnfRaisedErr err1

  -- Evaluates the biases
  err2 <- evalIO $ tryEvalRnf (FullyConnected (FullyConnected' (error expectedErrStr) w) mkListStore)
  assert $ rnfRaisedErr err2

  -- Evaluates the weights
  err3 <- evalIO $ tryEvalRnf (FullyConnected (FullyConnected' b (error expectedErrStr)) mkListStore)
  assert $ rnfRaisedErr err3
  
  OpaqueFullyConnected (fcValid :: FullyConnected i o) <- blindForAll genOpaqueFullyConnected
  noErrStr <- evalIO $ tryEvalRnf fcValid
  assert $ rnfNoError noErrStr

prop_fully_connected_is_randomizable :: Property
prop_fully_connected_is_randomizable = withTests 1 $ property $ do
  gen <- evalIO create
  _ :: FullyConnected 3 5 <- evalIO $ createRandomWith UniformInit gen
  success

prop_fully_connected_serialized :: Property
prop_fully_connected_serialized = property $ do
  OpaqueFullyConnected (fc :: FullyConnected i o) <- blindForAll genOpaqueFullyConnected
  let bs = encode fc
  let dec = decode bs :: Either String (FullyConnected i o)
  assert $ isRight dec
  let Right fc' = dec
      (FullyConnected (FullyConnected' b w)   _) = fc
      (FullyConnected (FullyConnected' b' w') _) = fc'
  H.extract b === H.extract b'
  H.extract w === H.extract w'

prop_fully_connected_show :: Property
prop_fully_connected_show = withTests 1 $ property $ do
  OpaqueFullyConnected fc <- blindForAll genOpaqueFullyConnected
  (show fc) `seq` success

prop_fully_connected_update :: Property
prop_fully_connected_update = withTests 1 $ property $ do
  let fc :: FullyConnected 3 5 = createFixed
      input = createFixedInput :: R 3
      (tape, out :: S ('D1 5)) = runForwards fc (S1D input)
      (grad, _   :: S ('D1 3)) = runBackwards fc tape out
      FullyConnected (FullyConnected' b w)   _ = runUpdate defSGD  fc grad
      FullyConnected (FullyConnected' b' w') _ = runUpdate defAdam fc grad
      expW :: H.L 5 3  = H.fromList [ 0.1214, 0.2429, 0.3643
                                    , 0.5229, 0.6171, 0.7114
                                    , 0.9243, 0.9914, 1.0586
                                    , 1.3257, 1.3657, 1.4057
                                    , 1.7271, 1.7400, 1.7529 ]
      expW' :: H.L 5 3 = H.fromList [ 0.1419, 0.2847, 0.4276
                                    , 0.5704, 0.7133, 0.8561
                                    , 0.9990, 1.1419, 1.2847
                                    , 1.4276, 1.5704, 1.7133
                                    , 1.8561, 1.9990, 2.1419 ]
  assert $ allCloseV b  (H.fromList [0.1214,0.2371,0.3529,0.4686,0.5843])
  assert $ allCloseV b' (H.fromList [0.1419,0.2847,0.4276,0.5704,0.7133] )
  assert $ allCloseP (S2D w)  (S2D expW) 0.001
  assert $ allCloseP (S2D w') (S2D expW') 0.001
  where
    createFixed :: forall i o. (KnownNat i, KnownNat o) => FullyConnected i o
    createFixed = fc
      where
        i' = fromIntegral $ natVal (Proxy :: Proxy i)
        o' = fromIntegral $ natVal (Proxy :: Proxy o)
        bias :: H.R o   = H.fromList (map (/7) [1..o'])
        acts :: H.L o i = H.fromList (map (/7) [1..(o' * i')])
        fc :: FullyConnected i o = FullyConnected (FullyConnected' bias acts) mkListStore

    createFixedInput :: forall i. (KnownNat i) => R i
    createFixedInput = H.fromList [1..i']
      where
          i' = fromIntegral $ natVal (Proxy :: Proxy i)

prop_fully_connected_foldable_gradient :: Property
prop_fully_connected_foldable_gradient = property $ do
  OpaqueFullyConnected (fc :: FullyConnected i o) <- blindForAll genOpaqueFullyConnected
  m <- forAll $ Gen.double (Range.constant 0 0.9)
  let FullyConnected fc'@(FullyConnected' b w) _ = fc
      FullyConnected' b' w' = mapGradient (*m) fc'
      bl  = D.toList $ H.extract b
      wl  = concat $ D.toLists $ H.extract w
      bl' = D.toList $ H.extract b'
      wl' = concat $ D.toLists $ H.extract w'
      [sb, sw] = squaredSums fc'
      ssum =  sum . map (** 2)
  assert $ allCloseL (map (*m) bl) bl'
  assert $ allCloseL (map (*m) wl) wl'
  assert $ (<= 0.001) $ abs $ (ssum bl) - sb
  assert $ (<= 0.001) $ abs $ (ssum wl) - sw

tests :: IO Bool
tests = checkParallel $$(discover)
