module Grenade.Assets.Paths where

import Paths_grenade

{-|
This module handles locating files on disk for networks
which are bundled with Grenade, irrespective of the storage
format.
-}

data BundledModel = MNIST | ResNet50

getPathForNetwork :: BundledModel -> IO FilePath
getPathForNetwork MNIST = getDataFileName "assets/mnistModel"
getPathForNetwork ResNet50 = undefined