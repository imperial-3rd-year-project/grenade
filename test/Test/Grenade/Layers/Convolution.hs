{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}

module Test.Grenade.Layers.Convolution where

import           Data.Constraint
import           Data.Proxy
import           Data.Singletons
import           Data.Singletons.Prelude.Num            ((%*))
import           GHC.TypeLits
import           Unsafe.Coerce

#if MIN_VERSION_singletons(2,6,0)
import           Data.Singletons.TypeLits               (SNat (..))
#endif

#if MIN_VERSION_base(4,9,0)
import           Data.Kind                              (Type)
#endif

import           Hedgehog
import qualified Hedgehog.Gen                           as Gen

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix
import           Test.Hedgehog.TypeLits

import qualified Numeric.LinearAlgebra                  as LA
import qualified Numeric.LinearAlgebra.Static           as H

import           Grenade.Core
import           Grenade.Layers.Convolution
import           Grenade.Types
import           Grenade.Utils.ListStore

import qualified Test.Grenade.Layers.Internal.Reference as Reference

data OpaqueConvolution :: Type where
     OpaqueConvolution :: Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns -> OpaqueConvolution

instance Show OpaqueConvolution where
    show (OpaqueConvolution n) = show n

data OpaqueBiasConvolution :: Type where
     OpaqueBiasConvolution :: Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns -> OpaqueBiasConvolution

instance Show OpaqueBiasConvolution where
    show (OpaqueBiasConvolution n) = show n

genConvolution :: ( KnownNat channels
                  , KnownNat filters
                  , KnownNat kernelRows
                  , KnownNat kernelColumns
                  , KnownNat strideRows
                  , KnownNat strideColumns
                  , KnownNat kernelFlattened
                  , kernelFlattened ~ (kernelRows * kernelColumns * channels)
                  ) => Gen (Convolution 'WithoutBias padding channels filters kernelRows kernelColumns strideRows strideColumns)
genConvolution = Convolution <$> uniformSample <*> pure mkListStore

genBiasConvolution :: ( KnownNat channels
                      , KnownNat filters
                      , KnownNat kernelRows
                      , KnownNat kernelColumns
                      , KnownNat strideRows
                      , KnownNat strideColumns
                      , KnownNat kernelFlattened
                      , kernelFlattened ~ (kernelRows * kernelColumns * channels)
                      ) => Gen (Convolution 'WithBias padding channels filters kernelRows kernelColumns strideRows strideColumns)
genBiasConvolution = BiasConvolution <$> uniformSample <*> randomVector <*> pure mkListStore

genOpaqueConvolution :: Gen OpaqueConvolution
genOpaqueConvolution = do
    channels <- genNat
    filters  <- genNat
    kernel_h <- genNat
    kernel_w <- genNat
    stride_h <- genNat
    stride_w <- genNat
    case (channels, filters, kernel_h, kernel_w, stride_h, stride_w) of
       ( SomeNat (pch :: Proxy ch), SomeNat  (_   :: Proxy fl),
         SomeNat (pkr :: Proxy kr), SomeNat  (pkc :: Proxy kc),
         SomeNat (_   :: Proxy sr), SomeNat  (_   :: Proxy sc)) ->
          let p1 = singByProxy pkr
              p2 = singByProxy pkc
              p3 = singByProxy pch
          in  case p1 %* p2 %* p3 of
            SNat -> OpaqueConvolution <$> (genConvolution :: Gen (Convolution 'WithoutBias padding ch fl kr kc sr sc))

genOpaqueBiasConvolution :: Gen OpaqueBiasConvolution
genOpaqueBiasConvolution = do
    channels <- genNat
    filters  <- genNat
    kernel_h <- genNat
    kernel_w <- genNat
    stride_h <- genNat
    stride_w <- genNat
    case (channels, filters, kernel_h, kernel_w, stride_h, stride_w) of
       ( SomeNat (pch :: Proxy ch), SomeNat  (_   :: Proxy fl),
         SomeNat (pkr :: Proxy kr), SomeNat  (pkc :: Proxy kc),
         SomeNat (_   :: Proxy sr), SomeNat  (_   :: Proxy sc)) ->
          let p1 = singByProxy pkr
              p2 = singByProxy pkc
              p3 = singByProxy pch
          in  case p1 %* p2 %* p3 of
            SNat -> OpaqueBiasConvolution <$> (genBiasConvolution :: Gen (Convolution 'WithBias padding ch fl kr kc sr sc))

prop_conv_no_padding_forward_and_backward_correct :: Property
prop_conv_no_padding_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueConvolution (convLayer'@(Convolution kernels _) :: Convolution 'WithoutBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueConvolution

  let convLayer = unsafeCoerce convLayer' :: Convolution 'WithoutBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols
      ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  -- calculate output size
  let rr = ((er - kr) `div` sr) + 1
      rc = ((ec - kc) `div` sc) + 1

  case (someNatVal er, someNatVal ec, someNatVal rr, someNatVal rc) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)), Just (SomeNat (pour :: Proxy outRows)), Just (SomeNat (_ :: Proxy outCols))) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy pour %* singByProxy (Proxy :: Proxy filters)
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideRows * (outRows - 1) <= (inRows - kernelRows + 0) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inRows - kernelRows + 0) <= (outRows * strideRows ) - 1 ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideCols * (outCols - 1) <= (inCols - kernelCols + 0) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inCols - kernelCols + 0) <= (outCols * strideCols ) - 1 ) ) ) of
                    (SNat, SNat, Dict, Dict, Dict, Dict) -> do
                      input :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let (tape, out@(S3D out') :: S ('D3 outRows outCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.convForwards input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc)

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (Convolution' dW, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw) = Reference.convBackProp input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc)

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw

prop_conv_same_upper_padding_forward_and_backward_correct :: Property
prop_conv_same_upper_padding_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueConvolution (convLayer'@(Convolution kernels _) :: Convolution 'WithoutBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueConvolution

  let convLayer = unsafeCoerce convLayer' :: Convolution 'WithoutBias 'SameUpper channels filters kernelRows kernelCols strideRows strideCols
      ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  annotate $ show (kr, kc, sr, sc, fs, cs)

  -- calculate padding size
  let pw = fromIntegral $ (ec - 1) * sc + kc - ec
      ph = fromIntegral $ (er - 1) * sr + kr - er
      padt = div ph 2
      padl = div pw 2
      padb = ph - padt
      padr = pw - padl

  case (someNatVal er, someNatVal ec) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)) ) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy pinr %* singByProxy (Proxy :: Proxy filters) ) of
                    (SNat, SNat) -> do
                      input@(S3D inp') :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let (tape, out@(S3D out') :: S ('D3 inRows inCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.convForwardsWithPadding input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (Convolution' dW, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw)   = Reference.convBackPropWithPadding (H.extract inp') cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw

prop_conv_same_lower_padding_forward_and_backward_correct :: Property
prop_conv_same_lower_padding_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueConvolution (convLayer'@(Convolution kernels _) :: Convolution 'WithoutBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueConvolution

  let convLayer = unsafeCoerce convLayer' :: Convolution 'WithoutBias 'SameLower channels filters kernelRows kernelCols strideRows strideCols
      ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  annotate $ show (kr, kc, sr, sc, fs, cs)

  -- calculate padding size
  let pw = fromIntegral $ (ec - 1) * sc + kc - ec
      ph = fromIntegral $ (er - 1) * sr + kr - er
      padb = div ph 2
      padr = div pw 2
      padt = ph - padb
      padl = pw - padr

  case (someNatVal er, someNatVal ec) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)) ) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy pinr %* singByProxy (Proxy :: Proxy filters) ) of
                    (SNat, SNat) -> do
                      input@(S3D inp') :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let (tape, out@(S3D out') :: S ('D3 inRows inCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.convForwardsWithPadding input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (Convolution' dW, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw)   = Reference.convBackPropWithPadding (H.extract inp') cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw

prop_conv_explicit_padding_forward_and_backward_correct :: Property
prop_conv_explicit_padding_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueConvolution (convLayer'@(Convolution kernels _) :: Convolution 'WithoutBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueConvolution

  let ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  padl <- forAll $ choose 0 kc
  padt <- forAll $ choose 0 kr
  padr <- forAll $ choose 0 kc
  padb <- forAll $ choose 0 kr

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  let rr = ((padt + er + padb - kr) `div` sr) + 1
      rc = ((padl + ec + padr - kc) `div` sc) + 1

  case (someNatVal er, someNatVal ec, someNatVal rr, someNatVal rc, someNatVal padl, someNatVal padt, someNatVal padr, someNatVal padb) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)), Just (SomeNat (poutr :: Proxy outRows)), Just (SomeNat (_  :: Proxy outCols)), Just (SomeNat (_  :: Proxy padl)), Just (SomeNat (_  :: Proxy padt)), Just (SomeNat (_  :: Proxy padr)), Just (SomeNat (_  :: Proxy padb)) ) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy poutr %* singByProxy (Proxy :: Proxy filters)
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideRows * (outRows - 1) <= (inRows - kernelRows + (padt + padb)) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inRows - kernelRows + (padt + padb)) <= (outRows * strideRows ) - 1 ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideCols * (outCols - 1) <= (inCols - kernelCols + (padl + padr)) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inCols - kernelCols + (padl + padr)) <= (outCols * strideCols ) - 1 ) ) ) of
                    (SNat, SNat, Dict, Dict, Dict, Dict) -> do
                      input@(S3D inp') :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let convLayer = unsafeCoerce convLayer' :: Convolution 'WithoutBias ('Padding padl padt padr padb) channels filters kernelRows kernelCols strideRows strideCols

                      let (tape, out@(S3D out') :: S ('D3 outRows outCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.convForwardsWithPadding input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc) (fromIntegral padl) (fromIntegral padt) (fromIntegral padr) (fromIntegral padb)

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (Convolution' dW, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw)   = Reference.convBackPropWithPadding (H.extract inp') cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc) (fromIntegral padl) (fromIntegral padt) (fromIntegral padr) (fromIntegral padb)

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw

prop_bias_conv_no_padding_forward_and_backward_correct :: Property
prop_bias_conv_no_padding_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueBiasConvolution (convLayer'@(BiasConvolution kernels bias _) :: Convolution 'WithBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueBiasConvolution

  let convLayer = unsafeCoerce convLayer' :: Convolution 'WithBias 'NoPadding channels filters kernelRows kernelCols strideRows strideCols
      ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  -- calculate output size
  let rr = ((er - kr) `div` sr) + 1
      rc = ((ec - kc) `div` sc) + 1

  case (someNatVal er, someNatVal ec, someNatVal rr, someNatVal rc) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)), Just (SomeNat (pour :: Proxy outRows)), Just (SomeNat (_ :: Proxy outCols))) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy pour %* singByProxy (Proxy :: Proxy filters)
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideRows * (outRows - 1) <= (inRows - kernelRows + 0) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inRows - kernelRows + 0) <= (outRows * strideRows ) - 1 ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideCols * (outCols - 1) <= (inCols - kernelCols + 0) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inCols - kernelCols + 0) <= (outCols * strideCols ) - 1 ) ) ) of
                    (SNat, SNat, Dict, Dict, Dict, Dict) -> do
                      input@(S3D inp') :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let (tape, out@(S3D out') :: S ('D3 outRows outCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.biasConvForwards input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (H.extract bias) (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc)

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (BiasConvolution' dW db, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw, refdb) = Reference.biasConvBackProp (H.extract inp') cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc)

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw
                      H.extract db `isSimilarVectorTo` refdb

prop_bias_conv_same_upper_forward_and_backward_correct :: Property
prop_bias_conv_same_upper_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueBiasConvolution (convLayer'@(BiasConvolution kernels bias _) :: Convolution 'WithBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueBiasConvolution

  let convLayer = unsafeCoerce convLayer' :: Convolution 'WithBias 'SameUpper channels filters kernelRows kernelCols strideRows strideCols
      ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  -- calculate padding size
  let pw = fromIntegral $ (ec - 1) * sc + kc - ec
      ph = fromIntegral $ (er - 1) * sr + kr - er
      padt = div ph 2
      padl = div pw 2
      padb = ph - padt
      padr = pw - padl

  case (someNatVal er, someNatVal ec) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)) ) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy pinr %* singByProxy (Proxy :: Proxy filters) ) of
                    (SNat, SNat) -> do
                      input@(S3D inp') :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let (tape, out@(S3D out') :: S ('D3 inRows inCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.biasConvForwardsWithPadding input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (H.extract bias) (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (BiasConvolution' dW db, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw, refdb) = Reference.biasConvBackPropWithPadding (H.extract inp') cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw
                      H.extract db `isSimilarVectorTo` refdb

prop_bias_conv_same_lower_forward_and_backward_correct :: Property
prop_bias_conv_same_lower_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueBiasConvolution (convLayer'@(BiasConvolution kernels bias _) :: Convolution 'WithBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueBiasConvolution

  let convLayer = unsafeCoerce convLayer' :: Convolution 'WithBias 'SameLower channels filters kernelRows kernelCols strideRows strideCols
      ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  -- calculate padding size
  let pw = fromIntegral $ (ec - 1) * sc + kc - ec
      ph = fromIntegral $ (er - 1) * sr + kr - er
      padb = div ph 2
      padr = div pw 2
      padt = ph - padb
      padl = pw - padr

  case (someNatVal er, someNatVal ec) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)) ) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy pinr %* singByProxy (Proxy :: Proxy filters) ) of
                    (SNat, SNat) -> do
                      input@(S3D inp') :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let (tape, out@(S3D out') :: S ('D3 inRows inCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.biasConvForwardsWithPadding input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (H.extract bias) (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (BiasConvolution' dW db, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw, refdb) = Reference.biasConvBackPropWithPadding (H.extract inp') cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral er) (fromIntegral ec) (fromIntegral sr) (fromIntegral sc) padl padt padr padb

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw
                      H.extract db `isSimilarVectorTo` refdb

prop_bias_conv_explicit_padding_forward_and_backward_correct :: Property
prop_bias_conv_explicit_padding_forward_and_backward_correct = withTests 20 $ property $ do
  OpaqueBiasConvolution (convLayer'@(BiasConvolution kernels bias _) :: Convolution 'WithBias padding channels filters kernelRows kernelCols strideRows strideCols) <- forAll genOpaqueBiasConvolution

  let ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
      kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
      kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
      sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
      sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)
      fs = fromIntegral $ natVal (Proxy :: Proxy filters   ) :: Int
      cs = fromIntegral $ natVal (Proxy :: Proxy channels  ) :: Int

  padl <- forAll $ choose 0 kc
  padt <- forAll $ choose 0 kr
  padr <- forAll $ choose 0 kc
  padb <- forAll $ choose 0 kr

  -- generate input size
  er <- forAll (Gen.element (ok sr kr))
  ec <- forAll (Gen.element (ok sc kc))

  let rr = ((padt + er + padb - kr) `div` sr) + 1
      rc = ((padl + ec + padr - kc) `div` sc) + 1

  case (someNatVal er, someNatVal ec, someNatVal rr, someNatVal rc, someNatVal padl, someNatVal padt, someNatVal padr, someNatVal padb) of
      ( Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)), Just (SomeNat (poutr :: Proxy outRows)), Just (SomeNat (_  :: Proxy outCols)), Just (SomeNat (_  :: Proxy padl)), Just (SomeNat (_  :: Proxy padt)), Just (SomeNat (_  :: Proxy padr)), Just (SomeNat (_  :: Proxy padb)) ) ->
        case ( singByProxy pinr %* singByProxy (Proxy :: Proxy channels)
             , singByProxy poutr %* singByProxy (Proxy :: Proxy filters)
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideRows * (outRows - 1) <= (inRows - kernelRows + (padt + padb)) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inRows - kernelRows + (padt + padb)) <= (outRows * strideRows ) - 1 ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( strideCols * (outCols - 1) <= (inCols - kernelCols + (padl + padr)) ) )
             , (unsafeCoerce (Dict :: Dict ()) :: Dict ( (inCols - kernelCols + (padl + padr)) <= (outCols * strideCols ) - 1 ) ) ) of
                    (SNat, SNat, Dict, Dict, Dict, Dict) -> do
                      input@(S3D inp') :: S ('D3 inRows inCols channels) <- forAll (S3D <$> uniformSample)

                      let convLayer = unsafeCoerce convLayer' :: Convolution 'WithBias ('Padding padl padt padr padb) channels filters kernelRows kernelCols strideRows strideCols

                      let (tape, out@(S3D out') :: S ('D3 outRows outCols filters)) = runForwards convLayer input
                          input'   = (\(S3D x) -> H.extract x) input
                          refOut   = Reference.biasConvForwardsWithPadding input' cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) (H.extract bias) (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc) (fromIntegral padl) (fromIntegral padt) (fromIntegral padr) (fromIntegral padb)

                      H.extract out' `isSimilarMatrixTo` refOut

                      let (BiasConvolution' dW db, S3D dx :: S ('D3 inRows inCols channels))
                            = runBackwards convLayer tape out
                          (refdx, refdw, refdb)   = Reference.biasConvBackPropWithPadding (H.extract inp') cs (fromIntegral er) (fromIntegral ec) (H.extract kernels) fs (fromIntegral kr) (fromIntegral kc) refOut (fromIntegral rr) (fromIntegral rc) (fromIntegral sr) (fromIntegral sc) (fromIntegral padl) (fromIntegral padt) (fromIntegral padr) (fromIntegral padb)

                      H.extract dx `isSimilarMatrixTo` refdx
                      H.extract dW `isSimilarMatrixTo` refdw
                      H.extract db `isSimilarVectorTo` refdb

prop_bias_convolution_same_as_torch :: Property
prop_bias_convolution_same_as_torch = withTests 1 $ property $ H.extract output `isSimilarMatrixTo` refOutput
  where
    refOutput  = LA.fromLists . concat $ outputList :: LA.Matrix RealNum
    S3D output = snd $ runForwards layer input :: S ('D3 4 4 3)

    layer   = BiasConvolution filters bias mkListStore :: Convolution 'WithBias 'NoPadding 1 3 4 4 2 2
    filters = H.tr . H.fromList . concat . concat $ filtersList
    bias    = H.fromList biasList
    input   = S2D . H.fromList . concat $ inputList :: S ('D2 10 10)

    inputList :: [[RealNum]]
    inputList = [[-1.8798149824142456,  0.1028530076146126,  1.8676880598068237,
                  -0.0617060922086239, -0.8356814980506897,  0.6487482190132141,
                   0.8218213915824890, -0.7828793525695801, -0.8224498033523560,
                  -0.9689628481864929
                 ]
                ,[-0.1040988788008690,  1.4789071083068848,  1.6646575927734375,
                   0.0237357039004564, -0.5236989855766296,  0.6900013089179993,
                  -2.3320152759552002, -1.6244263648986816, -1.0726952552795410,
                  -0.4659616351127625
                 ]
                ,[ 0.0606177598237991,  1.2164361476898193, -1.0005422830581665,
                   2.0863819122314453, -1.5312120914459229,  0.4628386795520782,
                  -1.5035923719406128,  0.5512505769729614, -0.6526298522949219,
                   0.6975844502449036
                 ]
                ,[ 1.5345604419708252,  0.2354260832071304, -0.2829766273498535,
                  -1.1268494129180908,  0.1847444474697113, -0.6780525445938110,
                   0.9897040128707886,  0.1145756468176842,  0.5400218367576599,
                  -0.6138708591461182
                 ]
                ,[-1.0893152952194214, -0.0248561538755894,  0.0916035994887352,
                   0.4344462454319000, -2.2688574790954590,  0.5042927861213684,
                  -1.2440685033798218,  0.4743961393833160,  0.2804549634456635,
                   0.1743697375059128
                 ]
                ,[-0.2276446968317032, -1.5715813636779785, -0.2782900631427765,
                   0.2772432267665863,  0.5334178209304810, -0.3267174661159515,
                  -0.6893687248229980, -0.8983678221702576,  1.7771846055984497,
                   0.6515364050865173
                 ]
                ,[-0.8124530911445618,  1.0634243488311768,  0.0320023111999035,
                  -1.3274892568588257,  0.3162937462329865,  0.6602431535720825,
                   0.7445837259292603,  0.1550943553447723,  0.1773122400045395,
                   0.5073298215866089
                 ]
                ,[ 0.8331560492515564,  1.1417356729507446,  1.3021006584167480,
                  -1.2554157972335815, -0.4929245114326477,  0.7357274293899536,
                   0.4516215324401855,  0.9766013026237488,  0.4637960195541382,
                  -1.2788373231887817
                 ]
                ,[-0.1299438923597336,  0.7541506290435791, -0.6450557708740234,
                  -0.0040270495228469,  0.7661917805671692,  0.0933827608823776,
                   0.3463796675205231,  0.1497271358966827,  0.6318596005439758,
                  -1.9423550367355347
                 ]
                ,[ 0.3258689343929291,  0.5653873682022095, -0.1593302637338638,
                   0.5531405806541443, -0.4313830733299255,  0.9392206668853760,
                   1.1189296245574951,  0.9076719880104065, -0.3691771030426025,
                   2.3205711841583252
                 ]
                ]

    filtersList :: [[[RealNum]]]
    filtersList = [[[ 0.0762235820293427, -0.0693399608135223,  0.1232089996337891, -0.1839968860149384]
                   ,[-0.2333435118198395,  0.0558746755123138,  0.1272304952144623,  0.1712160110473633]
                   ,[ 0.2303934693336487,  0.0637164115905762,  0.0874992907047272, -0.2228358089923859]
                   ,[-0.2300853431224823, -0.1301925778388977,  0.1110948324203491, -0.1208976805210114]
                   ]
                  ,[[-0.1482009291648865, -0.1966925859451294, -0.0440416038036346, -0.0799964368343353]
                   ,[ 0.0987019836902618,  0.0791423320770264,  0.2103689610958099, -0.2410672008991241]
                   ,[ 0.0028940439224243,  0.0728487372398376,  0.0047851204872131,  0.0087434351444244]
                   ,[ 0.2424267232418060, -0.1152470409870148,  0.2046474218368530, -0.1154336035251617]
                   ]
                  ,[[-0.1728554666042328,  0.2191164791584015,  0.0494877994060516, -0.1877064704895020]
                   ,[-0.0716654360294342,  0.1852594912052155,  0.1268731951713562,  0.1372097134590149]
                   ,[ 0.2338261902332306,  0.0943823754787445, -0.2169064879417419, -0.1956105828285217]
                   ,[-0.0422836244106293,  0.1209872066974640,  0.1442286074161530, -0.2474469542503357]
                   ]
                  ]

    biasList :: [RealNum]
    biasList = [-0.1065079271793365,  0.1141500771045685, -0.0434862375259399]

    outputList :: [[[RealNum]]]
    outputList = [[[-0.4326016604900360, -0.5381838679313660, -0.8197864294052124, -0.3077610433101654]
                  ,[-1.4160603284835815, -0.8261992335319519, -1.2307931184768677, -0.6055138707160950]
                  ,[-0.1438199430704117, -0.8861438035964966, -0.8439695835113525,  0.2755392491817474]
                  ,[-0.4444667994976044, -0.7047327160835266,  0.4867877364158630, -0.4625016748905182]
                  ]
                 ,[[ 1.2657278776168823,  0.0472350381314754,  0.3799144029617310,  0.2348928004503250]
                  ,[ 0.1615301817655563,  0.0406027995049953,  0.5752753019332886,  0.8285622000694275]
                  ,[ 0.5149132013320923,  0.4453703761100769,  0.2957187891006470,  0.5147784948348999]
                  ,[ 0.8481521010398865, -0.2448751479387283, -0.3152585625648499,  0.2738023698329926]
                  ]
                 ,[[ 1.0434051752090454, -0.3550303280353546,  0.0058417855761945, -0.6423963904380798]
                  ,[-1.1366062164306641,  0.8248907923698425, -0.0979268252849579, -0.2169524729251862]
                  ,[ 0.5547238588333130, -0.8525719046592712, -0.1231328472495079,  0.9879407286643982]
                  ,[ 0.8017534613609314, -1.2922899723052979,  0.5693746805191040, -0.4196736514568329]
                  ]
                 ]

tests :: IO Bool
tests = checkParallel $$(discover)
