{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Internal.Transpose where

import           Grenade.Layers.Internal.Transpose


import           Numeric.LinearAlgebra                  hiding (konst,
                                                         uniformSample, (===))

import           Hedgehog

prop_transpose4d_behaves_as_reference_swap_three_axis = withTests 1 $ property $ do
  let input           = (12 >< 4) [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 ]
      perms           = vector [2, 0, 1, 3]
      dims            = [2, 2, 3, 4]
      expected_output = (12 >< 4) [ 0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47 ]
      
      output = transpose4d dims perms input

  output === expected_output

prop_same_pad_pool_behaves_correctly_at_edges_three_channels = withTests 1 $ property $ do
  let input           = (12 >< 4) [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 ]
      perms           = vector [3, 2, 1, 0]
      dims            = [2, 2, 3, 4]
      expected_output = (24 >< 2) [ 0, 24, 12, 36, 4, 28, 16, 40, 8, 32, 20, 44, 1, 25, 13, 37, 5, 29, 17, 41, 9, 33, 21, 45, 2, 26, 14, 38, 6, 30, 18, 42, 10, 34, 22, 46, 3, 27, 15, 39, 7, 31, 19, 43, 11, 35, 23, 47 ]
      
      output = transpose4d dims perms input

  output === expected_output

tests :: IO Bool
tests = checkParallel $$(discover)
