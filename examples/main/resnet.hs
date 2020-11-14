{-# LANGUAGE CPP                       #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeFamilies              #-}

import           Control.Applicative
import           Control.Monad.Random
import           Control.Monad.Trans.Except

import qualified Data.Attoparsec.Text       as A
import qualified Data.Text                  as T
import qualified Data.Text.IO               as T
import qualified Data.Vector.Storable       as V

import           Options.Applicative

import           Grenade
import           Grenade.Networks.ResNet18


data ResNetOptions = ResNetOptions FilePath          -- Training/Test data


mnist' :: Parser ResNetOptions
mnist' = ResNetOptions <$> strOption (long "path")

main :: IO ()
main = do
    ResNetOptions path <- execParser (info (mnist' <**> helper) idm)
    res <- loadResNet path
    
    case res of 
      Nothing -> putStrLn "fail"
      Just x  -> putStrLn "nice"

