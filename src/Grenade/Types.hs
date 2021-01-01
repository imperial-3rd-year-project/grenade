{-# LANGUAGE CPP #-}
module Grenade.Types where

import Data.Typeable


#ifdef USE_FLOAT
import GHC.Float

type RealNum = Float       -- when using the hmatrix-float package

doubleToRealNum :: Double -> RealNum
doubleToRealNum = double2Float

realNumToDouble :: RealNum -> Double
realNumToDouble = float2Double

#else

#ifdef USE_DOUBLE
type RealNum = Double   -- when using the hmatrix package

doubleToRealNum :: Double -> RealNum
doubleToRealNum = id

realNumToDouble :: RealNum -> Double
realNumToDouble = id
#else

#ifdef FLYCHECK
type RealNum = Double

doubleToRealNum :: Double -> RealNum
doubleToRealNum = id

realNumToDouble :: RealNum -> Double
realNumToDouble = id

#else
You have to provide the preprocessor directive (for GHC and GCC) -DUSE_FLOAT or -DUSE_DOUBLE

#endif
#endif
#endif

-- | String representation of type `F`, which is either Float or Double type.
nameF :: String
nameF = show (typeRep (Proxy :: Proxy RealNum))
