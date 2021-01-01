{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Grenade.Utils.Symbols (ValidRealNum) where

import GHC.TypeLits
import Data.Type.Bool
import Data.Kind (Constraint)
import Data.Symbol.Ascii (ToList)

-- | Constraint for symbols representing RealNums
type family ValidRealNum (x :: Symbol) :: Constraint where
  ValidRealNum x = ParseRealNum x ~ 'True

-- | Parse a number with an optional decimal point
type family ParseRealNum (sym :: Symbol) :: Bool where
  ParseRealNum sym = ParseRealNum1 sym (ToList sym)

type family ParseRealNum1 (orig :: Symbol) (sym :: [Symbol]) :: Bool where
  ParseRealNum1 _ '[]        = TypeError ('Text "Parse error: empty string")
  ParseRealNum1 orig '["."]     = TypeError ('Text "Parse error: invalid form for value, try '0' instead in " ':<>: 'ShowType orig)
  ParseRealNum1 orig ("." ': _) = TypeError ('Text "Parse error: invalid form for value, try prepending a zero in " ':<>: 'ShowType orig)
  ParseRealNum1 orig xs      = ParseRealNum2 orig xs 0

type family ParseRealNum2 (orig :: Symbol) (sym :: [Symbol]) (c :: Nat)  :: Bool where
  -- If we encounter more than 1 decimal point, raise an error
  ParseRealNum2 orig _ 2             = TypeError ('Text "Parse error: too many decimal points in " ':<>: 'ShowType orig)
  -- If the last character is a decimal point, then read won't parse this, so raise an error
  ParseRealNum2 orig '["."] _        = TypeError ('Text "Parse error: invalid form for value, try removing the decimal point in " ':<>: 'ShowType orig)
  ParseRealNum2 _ '[] _              = 'True
  -- If we see a decimal point, increment the counter
  ParseRealNum2 orig ((".") ': xs) c = ParseRealNum2 orig xs (c + 1)
  -- Check that the current character is a digit and then parse the rest
  ParseRealNum2 orig (x ': xs)     c = (IsDigit orig x) && (ParseRealNum2 orig xs c)

type family IsDigit (orig :: Symbol) (sym :: Symbol) :: Bool where
  IsDigit _ "0" = 'True
  IsDigit _ "1" = 'True
  IsDigit _ "2" = 'True
  IsDigit _ "3" = 'True
  IsDigit _ "4" = 'True
  IsDigit _ "5" = 'True
  IsDigit _ "6" = 'True
  IsDigit _ "7" = 'True
  IsDigit _ "8" = 'True
  IsDigit _ "9" = 'True
  IsDigit orig other = 
    TypeError ('Text "Parse error: "
               ':<>: 'ShowType other
               ':<>: 'Text " is not a valid digit in "
               ':<>: 'ShowType orig)
