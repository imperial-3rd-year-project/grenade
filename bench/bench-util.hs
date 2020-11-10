{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}

import           Criterion.Main

import           Grenade
import           Grenade.Utils.LinearAlgebra

main :: IO ()
main = do
  let batchBenchSize1 = 32
      batchBenchSize2 = 128

  xs1D :: [S ('D1 5)]   <- sequence $ take batchBenchSize1 $ repeat randomOfShape
  ys1D :: [S ('D1 250)] <- sequence $ take batchBenchSize1 $ repeat randomOfShape

  xs'1D :: [S ('D1 5)]   <- sequence $ take batchBenchSize2 $ repeat randomOfShape
  ys'1D :: [S ('D1 250)] <- sequence $ take batchBenchSize2 $ repeat randomOfShape

  xs2D :: [S ('D2 3 4)]     <- sequence $ take batchBenchSize1 $ repeat randomOfShape
  ys2D :: [S ('D2 100 100)] <- sequence $ take batchBenchSize1 $ repeat randomOfShape

  xs'2D :: [S ('D2 3 4)]     <- sequence $ take batchBenchSize2 $ repeat randomOfShape
  ys'2D :: [S ('D2 100 100)] <- sequence $ take batchBenchSize2 $ repeat randomOfShape

  defaultMain
    [ bgroup
        "bmean 1D"
        [ bench "bmean 32 of 5" $ whnf bmean xs1D
        , bench "bmean 32 of 250" $ whnf bmean ys1D
        , bench "bmean 128 of 5" $ whnf bmean xs'1D
        , bench "bmean 128 of 250" $ whnf bmean ys'1D
        ]
    , bgroup
        "bmean 2D"
        [ bench "bmean 32 of 3x4" $ whnf bmean xs2D
        , bench "bmean 32 of 100x100" $ whnf bmean ys2D
        , bench "bmean 128 of 3x4" $ whnf bmean xs'2D
        , bench "bmean 128 of 100x100" $ whnf bmean ys'2D
        ]
    , bgroup
        "bvar 1D"
        [ bench "bvar 32 of 5" $ whnf bvar xs1D
        , bench "bvar 32 of 250" $ whnf bvar ys1D
        , bench "bvar 128 of 5" $ whnf bvar xs'1D
        , bench "bvar 128 of 250" $ whnf bvar ys'1D
        ]
    , bgroup
        "bvar 2D"
        [ bench "bvar 32 of 3x4" $ whnf bvar xs2D
        , bench "bvar 32 of 100x100" $ whnf bvar ys2D
        , bench "bvar 128 of 3x4" $ whnf bvar xs'2D
        , bench "bvar 128 of 100x100" $ whnf bvar ys'2D
        ]
    ]

