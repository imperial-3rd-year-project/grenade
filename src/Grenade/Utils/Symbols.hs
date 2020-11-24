{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Grenade.Utils.Symbols (ValidDouble) where

import GHC.TypeLits
import Data.Type.Bool
import Data.Kind (Constraint)
import Data.Symbol.Ascii (ToList)

-- | Constraint for symbols representing doubles
type family ValidDouble (x :: Symbol) :: Constraint where
  ValidDouble x = ParseDouble x ~ 'True

-- | Parse a number with an optional decimal point
type family ParseDouble (sym :: Symbol) :: Bool where
  ParseDouble sym = ParseDouble1 sym (ToList sym)

type family ParseDouble1 (orig :: Symbol) (sym :: [Symbol]) :: Bool where
  ParseDouble1 _ '[]        = TypeError ('Text "Parse error: empty string")
  ParseDouble1 orig '["."]     = TypeError ('Text "Parse error: invalid form for value, try '0' instead in " ':<>: 'ShowType orig)
  ParseDouble1 orig ("." ': _) = TypeError ('Text "Parse error: invalid form for value, try prepending a zero in " ':<>: 'ShowType orig)
  ParseDouble1 orig xs      = ParseDouble2 orig xs 0

type family ParseDouble2 (orig :: Symbol) (sym :: [Symbol]) (c :: Nat)  :: Bool where
  -- If we encounter more than 1 decimal point, raise an error
  ParseDouble2 orig _ 2             = TypeError ('Text "Parse error: too many decimal points in " ':<>: 'ShowType orig)
  -- If the last character is a decimal point, then read won't parse this, so raise an error
  ParseDouble2 orig '["."] _        = TypeError ('Text "Parse error: invalid form for value, try removing the decimal point in " ':<>: 'ShowType orig)
  ParseDouble2 _ '[] _              = 'True
  -- If we see a decimal point, increment the counter
  ParseDouble2 orig ((".") ': xs) c = ParseDouble2 orig xs (c + 1)
  -- Check that the current character is a digit and then parse the rest
  ParseDouble2 orig (x ': xs)     c = (IsDigit orig x) && (ParseDouble2 orig xs c)

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
