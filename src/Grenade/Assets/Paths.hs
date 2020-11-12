{-# LANGUAGE RankNTypes #-}

module Grenade.Assets.Paths where

import qualified Data.ByteString as BS
import           Data.Serialize               (Get, runGet, get, Serialize)
import           Paths_grenade


{-|
This module handles locating files on disk for networks
which are bundled with Grenade, irrespective of the storage
format, and loading networks.
-}

data BundledModel = MNIST | ResNet50

getPathForNetwork :: BundledModel -> IO FilePath
getPathForNetwork MNIST = getDataFileName "assets/mnistModel"
getPathForNetwork ResNet50 = undefined

loadNetwork :: forall network. (Serialize network) => FilePath -> IO network
loadNetwork path = do
  modelData <- BS.readFile path
  either fail return $ runGet (get :: (Serialize network) => Get network) modelData

