{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}

{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
module Test.Grenade.Layers.BatchNorm where

import           Control.Monad
import           Data.List                         (zipWith5)
import           Data.Proxy
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static      (L, R)
import qualified Numeric.LinearAlgebra.Static      as H

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

import           Grenade.Core
import           Grenade.Layers.BatchNormalisation
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

batchnorm :: forall channels rows columns momentum.
  (KnownNat channels, KnownNat rows, KnownNat columns, KnownNat momentum)
  => Bool -> R channels -> R channels -> R channels -> R channels -> BatchNorm channels rows columns momentum
batchnorm training gamma beta mean var =
  let ε      = 0.00001
  in BatchNorm training (BatchNormParams gamma beta) mean var ε mkListStore

prop_batchnorm_train_behaves_as_reference :: Property
prop_batchnorm_train_behaves_as_reference = property $ do
  height   :: Int <- forAll $ choose 2 100
  width    :: Int <- forAll $ choose 2 100
  channels :: Int <- forAll $ choose 1 100

  case (someNatVal (fromIntegral height), someNatVal (fromIntegral width), someNatVal (fromIntegral channels), channels) of
    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), _, 1) -> do
      inp :: S ('D2 h w) <- forAll genOfShape
      guard . not $ elementsEqual inp
      g :: R 1 <- forAll randomVector
      b :: R 1 <- forAll randomVector
      m :: R 1 <- forAll randomVector
      v :: R 1 <- forAll randomPositiveVector

      let layer   = batchnorm False g b m v :: BatchNorm 1 h w 90
          S2D out = snd $ runForwards layer inp :: S ('D2 h w)
          S2D ref = run2DBatchNorm layer inp :: S ('D2 h w)
      H.extract out === H.extract ref

    (Just (SomeNat (Proxy :: Proxy h)), Just (SomeNat (Proxy :: Proxy w)), Just (SomeNat (Proxy :: Proxy c)), _) -> do
      inp :: S ('D3 h w c) <- forAll genOfShape
      g :: R c <- forAll randomVector
      b :: R c <- forAll randomVector
      m :: R c <- forAll randomVector
      v :: R c <- forAll randomPositiveVector

      let layer   = batchnorm False g b m v :: BatchNorm c h w 90
          S3D out = snd $ runForwards layer inp :: S ('D3 h w c)
          S3D ref = run3DBatchNorm layer inp    :: S ('D3 h w c)
      H.extract out === H.extract ref

prop_batchnorm_1D_forward_same_as_torch :: Property
prop_batchnorm_1D_forward_same_as_torch = withTests 1 $ property $ do
    let g = H.fromList weight :: R 10
        b = H.fromList bias   :: R 10
        m = H.fromList mean   :: R 10
        v = H.fromList var    :: R 10

        bn = batchnorm False g b m v :: BatchNorm 10 4 4 90

        mat    = H.fromList . concat . concat $ input :: L 40 4
        x      = S3D mat :: S ('D3 4 4 10)
        y      = snd $ runForwards bn x :: S ('D3 4 4 10)

        mat' = H.fromList . concat . concat $ ref_out :: L 40 4

    assert $ allClose y (S3D mat') 
  where
    weight = [ -0.9323,  1.0161,  0.1728,  0.3656, -0.6816,  0.0334,  0.8494, -0.6669, -0.1527, -0.7004 ]
    bias   = [  0.7582,  1.0068, -0.2532, -1.5240, -1.0370,  0.8442,  0.5867, -1.2567,  0.4283, -0.0001 ]
    mean   = [ -0.4507,  0.9090, -1.4717,  0.5009,  0.8931, -0.4792,  0.0432,  0.4649, -0.6547, -1.3197 ]
    var    = [  1.2303,  1.4775,  0.8372,  0.1644,  0.9392,  0.2103,  0.4951,  0.2482,  0.7559,  0.3686 ]

    input  = [ [ [ -0.70681204, -0.20616523, -0.33806887, -1.52378976   ]
               , [ 0.056113367, -0.51263034, -0.28884589, -2.64030218   ]
               , [ -1.19894597, -1.16714501, -0.19816216, -1.13361239   ]
               , [ -0.81997509, -1.05715847,  0.59198695,  0.51939314   ] ]
             , [ [ -0.18338945, -1.08975303,  0.30558434,  0.85780441   ]
               , [ -0.47586514,  0.16499641,  2.18205571, -0.11155529   ]
               , [ 1.090167402,  0.92460924,  0.42982020,  1.30098605   ]
               , [ 0.286766794, -1.90825951, -0.91737461, -1.11035680   ] ]
             , [ [ 1.042808533,  0.08287286, -0.92343962, -0.49747768   ]
               , [ -0.21943949,  0.61554014, -2.25771808, -0.04292159   ]
               , [ 1.290057424, -1.07323992, -1.00024509,  1.30155622   ]
               , [ 0.472014425, -0.96431374,  0.77593171, -1.19090688   ] ]
             , [ [ 0.993361895,  0.82586401, -1.64278686,  1.25544464   ]
               , [ 0.239656539, -0.81472164,  1.32814168,  0.78350490   ]
               , [ -0.16597847,  0.74175131, -1.29834091, -1.28858852   ]
               , [ 1.307537318,  0.55525642, -0.04312540,  0.24699424   ] ]
             , [ [ 0.391699581, -0.09803850, -0.41061267,  0.34999904   ]
               , [ -2.22257169,  0.43748092, -1.21343314,  0.39576068   ]
               , [ 0.003147978, -1.00396716,  1.27623140,  1.17001295   ]
               , [ -0.58247902, -0.15453417, -0.37016496,  0.04613848   ] ]
             , [ [ 0.521356827,  0.94643139,  1.11394095,  0.60162323   ]
               , [ -0.90214585, -0.75316292,  2.20823979, -1.63446676   ]
               , [ 0.668517357,  0.62832462,  0.31174039, -0.04457542   ]
               , [ -0.24607617,  0.12855675, -1.62831199, -0.23100854   ] ]
             , [ [ -0.43619379, -0.41219231,  0.07910434, -0.20312546   ]
               , [ 1.670419093, -0.26496240, -1.53759109,  1.00907373   ]
               , [ -1.04028647, -1.37309467, -0.79040497, -0.15661381   ]
               , [ 0.009049783, -0.05525103,  1.44492578,  0.44786781   ] ]
             , [ [ 1.431640263, -0.12869687,  1.25025844,  0.07864278   ]
               , [ -1.69032764, -0.07707843,  0.11284181, -0.00826502   ]
               , [ -0.92387816, -0.83121442,  0.42292186, -0.49128937   ]
               , [ -1.62631051,  0.98236626, -1.69256067, -0.66552013   ] ]
             , [ [ 0.154654814,  0.59295737,  0.48604089,  0.46829459   ]
               , [ 0.624001921,  2.11190581, -1.80008912,  0.26847255   ]
               , [ -0.36086676,  0.94211035,  0.19112136, -0.04113261   ]
               , [ -0.94438538, -0.38932472, -0.29867526,  0.34307864   ] ]
             , [ [ 1.016388653, -0.41974341, -0.94618958,  0.22629515   ]
               , [ -2.04437517, -1.14956784,  0.38054388,  0.82105201   ]
               , [ 0.054255251,  1.03682625,  0.29021424, -0.42736151   ]
               , [ -0.00021907,  0.98816186,  0.23878140, -0.17728853   ] ] ]
    
    ref_out = [ [ [ 0.9734674096107483, 0.5526634454727173, 0.6635311841964722, 1.660154104232788]
                , [0.3322128653526306, 0.8102537393569946, 0.6221582293510437, 2.5986058712005615]
                , [1.3871161937713623, 1.360386848449707, 0.5459367036819458, 1.3322019577026367]
                , [1.068583369255066, 1.267940878868103, -0.11819997429847717, -0.05718337371945381] ]
              , [ [0.0936361625790596, -0.66402268409729, 0.5023852586746216, 0.9640040397644043]
                , [-0.15085376799106598, 0.3848632574081421, 2.070988893508911, 0.15368467569351196]
                , [1.1582437753677368, 1.019848346710205, 0.606238067150116, 1.334473967552185]
                , [0.4866550862789154, -1.3482388257980347, -0.5199258923530579, -0.6812460422515869] ]
              , [ [0.22167539596557617, 0.04038754478096962, -0.14965875446796417, -0.06921406835317612]
                , [-0.016705399379134178, 0.14098398387432098, -0.4016427993774414, 0.016630740836262703]
                , [0.2683693766593933, -0.17794916033744812, -0.16416378319263458, 0.2705409526824951]
                , [0.11387854814529419, -0.15737800300121307, 0.1712745875120163, -0.20017105340957642] ]
              , [ [-1.0799674987792969, -1.230993390083313, -3.456873655319214, -0.8436583876609802]
                , [-1.7595523595809937, -2.7102415561676025, -0.7781105041503906, -1.2691868543624878]
                , [-2.1252965927124023, -1.30683434009552, -3.146301031112671, -3.137507677078247]
                , [-0.7966886162757874, -1.4749890565872192, -2.0145251750946045, -1.7529363632202148] ]
              , [ [-0.6843588948249817, -0.3399200439453125, -0.1200828030705452, -0.655030369758606]
                , [1.1542901992797852, -0.7165574431419373, 0.44455069303512573, -0.6872150897979736]
                , [-0.41108575463294983, 0.29723069071769714, -1.306460976600647, -1.2317562103271484]
                , [0.0007928922423161566, -0.3001859486103058, -0.14853018522262573, -0.4413214921951294] ]
              , [ [0.9170715808868408, 0.9480301737785339, 0.9602301120758057, 0.9229174852371216]
                , [0.8133963942527771, 0.8242470026016235, 1.0399290323257446, 0.760060727596283]
                , [0.9277894496917725, 0.9248621463775635, 0.9018049836158752, 0.8758541345596313]
                , [0.8611786365509033, 0.88846355676651, 0.7605089545249939, 0.862276017665863] ]
              , [ [0.007999604567885399, 0.036973003298044205, 0.6300419569015503, 0.28934815526008606]
                , [2.5509982109069824, 0.21470165252685547, -1.3215526342391968, 1.7526549100875854]
                , [-0.7212311029434204, -1.1229807138442993, -0.4195865988731384, 0.34549471735954285]
                , [0.5454756021499634, 0.4678548276424408, 2.278794050216675, 1.0751949548721313] ]
              , [ [-2.550779104232788, -0.4621109068393707, -2.307981491088867, -0.7396559119224548]
                , [1.628288984298706, -0.5312073826789856, -0.7854347229003906, -0.6233210563659668]
                , [0.6023192405700684, 0.4782795310020447, -1.2005081176757812, 0.023255640640854836]
                , [1.542595624923706, -1.9493807554244995, 1.631278157234192, 0.2564810514450073] ]
              , [ [0.28615128993988037, 0.20917125046253204, 0.22794923186302185, 0.2310660481452942]
                , [0.20371884107589722, -0.05760492384433746, 0.6294671297073364, 0.2661612331867218]
                , [0.3766934275627136, 0.14784877002239227, 0.2797465920448303, 0.32053783535957336]
                , [0.4791780710220337, 0.381691575050354, 0.3657706081867218, 0.25305798649787903] ]
              , [ [-2.6950571537017822, -1.0383073091506958, -0.4309888482093811, -1.7835899591445923]
                , [0.8358993530273438, -0.19636774063110352, -1.9615343809127808, -2.469712972640991]
                , [-1.5851213932037354, -2.7186343669891357, -1.8573282957077026, -1.029518961906433]
                , [-1.5222787857055664, -2.66249418258667, -1.7979943752288818, -1.3180080652236938] ] ]


tests :: IO Bool
tests = checkParallel $$(discover)


-- REFERENCE FUNCTIONS

run2DBatchNorm :: forall h w m.
                  (KnownNat h, KnownNat w, KnownNat m)
               => BatchNorm 1 h w m -> S ('D2 h w) ->  S ('D2 h w)
run2DBatchNorm (BatchNorm False (BatchNormParams gamma beta) runningMean runningVar ε _) (S2D x)
  = let [m]    = vectorToList runningMean
        [v]    = vectorToList runningVar
        [g]    = vectorToList gamma
        [b]    = vectorToList beta
        std    = sqrt $ v + ε
        x_norm = H.dmmap (\a -> (a - m) / std) x
        out    = H.dmmap (\a -> g * a + b) x_norm
    in S2D out

run3DBatchNorm :: forall h w m c.
                  (KnownNat h, KnownNat w, KnownNat m, KnownNat c)
               => BatchNorm c h w m -> S ('D3 h w c) ->  S ('D3 h w c)
run3DBatchNorm (BatchNorm False (BatchNormParams gamma beta) runningMean runningVar ε _) inp
  = let ms     = vectorToList runningMean
        vs     = vectorToList runningVar
        gs     = vectorToList gamma
        bs     = vectorToList beta

        cs     = splitChannels inp :: [S ('D2 h w)]

        f c g b m v = let gs' = listToVector [g] :: R 1
                          bs' = listToVector [b] :: R 1
                          ms' = listToVector [m] :: R 1
                          vs' = listToVector [v] :: R 1
                          bn' = BatchNorm False (BatchNormParams gs' bs') ms' vs' ε undefined :: BatchNorm 1 h w m
                      in  run2DBatchNorm bn' c
      in combineChannels $ zipWith5 f cs gs bs ms vs
