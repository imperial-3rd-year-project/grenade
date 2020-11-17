{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}

import           Criterion.Main
import           Control.Monad  (replicateM)

import           Grenade
import           Grenade.Utils.LinearAlgebra

main :: IO ()
main = do
  let batchBenchSize1 = 32
      batchBenchSize2 = 128

  xs1D :: [S ('D1 5)]   <- replicateM batchBenchSize1 randomOfShape
  ys1D :: [S ('D1 250)] <- replicateM batchBenchSize1 randomOfShape

  xs'1D :: [S ('D1 5)]   <- replicateM batchBenchSize2 randomOfShape
  ys'1D :: [S ('D1 250)] <- replicateM batchBenchSize2 randomOfShape

  xs2D :: [S ('D2 3 4)]     <- replicateM batchBenchSize1 randomOfShape
  ys2D :: [S ('D2 100 100)] <- replicateM batchBenchSize1 randomOfShape

  xs'2D :: [S ('D2 3 4)]     <- replicateM batchBenchSize2 randomOfShape
  ys'2D :: [S ('D2 100 100)] <- replicateM batchBenchSize2 randomOfShape

  defaultMain
    [ bgroup
        "bmean 1D"
        [ bench "bmean 32 of 5"    $ nf bmean xs1D
        , bench "bmean 32 of 250"  $ nf bmean ys1D
        , bench "bmean 128 of 5"   $ nf bmean xs'1D
        , bench "bmean 128 of 250" $ nf bmean ys'1D
        ]
    , bgroup
        "bmean 2D"
        [ bench "bmean 32 of 3x4"      $ nf bmean xs2D
        , bench "bmean 32 of 100x100"  $ nf bmean ys2D
        , bench "bmean 128 of 3x4"     $ nf bmean xs'2D
        , bench "bmean 128 of 100x100" $ nf bmean ys'2D
        ]
    , bgroup
        "bvar 1D"
        [ bench "bvar 32 of 5"    $ nf bvar xs1D
        , bench "bvar 32 of 250"  $ nf bvar ys1D
        , bench "bvar 128 of 5"   $ nf bvar xs'1D
        , bench "bvar 128 of 250" $ nf bvar ys'1D
        ]
    , bgroup
        "bvar 2D"
        [ bench "bvar 32 of 3x4"      $ nf bvar xs2D
        , bench "bvar 32 of 100x100"  $ nf bvar ys2D
        , bench "bvar 128 of 3x4"     $ nf bvar xs'2D
        , bench "bvar 128 of 100x100" $ nf bvar ys'2D
        ]
    ]

