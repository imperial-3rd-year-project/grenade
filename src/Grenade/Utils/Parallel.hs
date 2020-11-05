{-# LANGUAGE DataKinds #-}
module Grenade.Utils.Parallel where

import Control.Parallel

zipToTuple :: (a -> b -> (x, y)) -> [a] -> [b] -> ([x], [y])
zipToTuple f as bs = zipToTuple' f as bs ([], [])
  where 
    zipToTuple' _ [] _ t = t
    zipToTuple' _ _ [] t = t 
    zipToTuple' f (a:as) (b:bs) (xs, ys)
      = let (x, y) = f a b 
        in  zipToTuple' f as bs (x:xs, y:ys)

parZipWith :: (a -> b -> c) -> [a] -> [b] -> [c] 
parZipWith _ [] _  = []
parZipWith _ _ []  = []
parZipWith f (x:xs) (y:ys) 
  = let fxy = f x y 
    in fxy `par` (fxy : parZipWith f xs ys)