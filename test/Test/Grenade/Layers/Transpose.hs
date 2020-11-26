{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing #-}

{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}



module Test.Grenade.Layers.Transpose where

import           Grenade.Core
import           Grenade.Layers.Transpose

import           Numeric.LinearAlgebra        hiding (R, konst, uniformSample,
                                               (===))
import           Numeric.LinearAlgebra.Static as H hiding ((===))

import           Hedgehog

prop_transpose_layer_calls_transpose_correctly = withTests 1 $ property $ do
  -- TEST 1
  let input :: L 12 4 = H.fromList [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 ]
      perms :: R 4    = H.fromList [ 2, 0, 1, 3 ]
      expected_output = (12 >< 4)  [ 0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47 ]

      input' = S4D input :: S ('D4 2 2 3 4)
      layer = Transpose perms :: Transpose 4 ('D4 2 2 3 4) ('D4 3 2 2 4)

      S4D output = snd $ runForwards layer input' :: S ('D4 3 2 2 4)

  H.extract output === expected_output

  -- TEST 2
  let input :: L 12 4 = H.fromList [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 ]
      perms :: R 4    = H.fromList [ 3, 2, 1, 0 ]
      expected_output = (24 >< 2)  [ 0, 24, 12, 36, 4, 28, 16, 40, 8, 32, 20, 44, 1, 25, 13, 37, 5, 29, 17, 41, 9, 33, 21, 45, 2, 26, 14, 38, 6, 30, 18, 42, 10, 34, 22, 46, 3, 27, 15, 39, 7, 31, 19, 43, 11, 35, 23, 47 ]

      input' = S4D input :: S ('D4 2 2 3 4)
      layer = Transpose perms :: Transpose 4 ('D4 2 2 3 4) ('D4 4 3 2 2)

      S4D output = snd $ runForwards layer input' :: S ('D4 4 3 2 2)

  H.extract output === expected_output

tests :: IO Bool
tests = checkParallel $$(discover)
