
{-# LANGUAGE CPP        #-}
{-# LANGUAGE RankNTypes #-}
module Test.Hedgehog.Compat (
    (...)
  , choose
  , blindForAll
  , genRealNum
  , isSimilarListTo
  , isSimilarListOfListsTo
  , isWithinOf

  -- Constants
  , precision
  , numericalGradDiff
  , numericalGradError
  ) where

import GHC.Stack (HasCallStack, withFrozenCallStack)

import           Hedgehog                   (Gen, MonadGen)
import qualified Hedgehog.Gen               as Gen
import           Hedgehog.Internal.Property
import qualified Hedgehog.Range             as Range

import           Grenade.Types

(...) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(...) = (.) . (.)
{-# INLINE (...) #-}

choose :: Integral a => a -> a -> Gen a
choose = Gen.integral ... Range.constant

blindForAll :: Monad m => Gen a -> PropertyT m a
blindForAll = forAllWith (const "blind")

#if USE_FLOAT
precision :: (Eq a, Ord a, Num a, Fractional a, Show a) => a
precision = 1e-2

numericalGradDiff :: (Num a, Fractional a) => a
numericalGradDiff = 0.0001

numericalGradError :: (Num a, Fractional a) => a
numericalGradError = 0.02

genRealNum :: MonadGen m => Range.Range RealNum -> m RealNum
genRealNum = Gen.float
#else 
precision :: (Eq a, Ord a, Num a, Fractional a, Show a) => a
precision = 1e-4

numericalGradDiff :: (Num a, Fractional a) => a
numericalGradDiff = 0.00001

numericalGradError :: (Num a, Fractional a) => a
numericalGradError = 1e-4

genRealNum :: MonadGen m => Range.Range RealNum -> m RealNum
genRealNum = Gen.double
#endif

isWithinOf :: (MonadTest m, HasCallStack, Fractional a, Show a, Ord a) => a -> a -> a -> m ()
isWithinOf n x y = 
  if abs (x - y) < n then
    success
  else
    withFrozenCallStack $
      failWith Nothing $ unlines [
          "━━━ Not within " ++ show n ++ " of each other ━━━"
          , show x
          , show y
          ]

isSimilarListTo :: (MonadTest m, HasCallStack, Fractional a, Ord a, Show a) => [a] -> [a] -> m ()
isSimilarListTo xs ys = 
  withFrozenCallStack $ 
    diff xs maxDiff ys
  
isSimilarListOfListsTo :: (MonadTest m, HasCallStack, Fractional a, Ord a, Show a) => [[a]] -> [[a]] -> m ()
isSimilarListOfListsTo xs ys = 
  withFrozenCallStack $ 
    diff xs maxDiff' ys
  where 
    -- | taxicab norm of each list
    maxDiff' (a:as) (b:bs) = maxDiff a b || maxDiff' as bs
    maxDiff' []     []     = True
    maxDiff' _      _      = False

-- | taxicab norm
maxDiff :: (Fractional a, Ord a, Show a) => [a] -> [a] -> Bool
maxDiff (a:as) (b:bs) | a - b < 0.000001 = maxDiff as bs 
maxDiff []     []     = True 
maxDiff _      _      = False