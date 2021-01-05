{-# LANGUAGE RankNTypes #-}
{-|
Module      : Grenade.Assets.Paths
Descripton  : Deals with files for networks bundled with Grenade

This module handles locating and downloading files for networks
which are bundled with Grenade, irrespective of the storage
format, and loading networks.

-}

module Grenade.Assets.Paths where

import qualified Data.ByteString as BS
import           Data.Serialize               (Get, runGet, get, Serialize)
import           System.Directory
import           System.FilePath.Posix
import           System.Process
import           System.IO
import           Control.Monad                (unless)

data BundledModel = MNIST | ResNet18 | TinyYoloV2 | SuperResolution

getPathForNetwork :: BundledModel -> IO FilePath
getPathForNetwork model = do
  configDir <- getXdgDirectory XdgData "grenade"
  createDirectoryIfMissing True configDir
  let fileName = case model of
                   MNIST           -> "mnistModel"
                   TinyYoloV2      -> "tiny-yolo-v2.onnx"
                   ResNet18        -> "resnet18-v1-7.onnx"
                   SuperResolution -> "super-resolution-10.onnx"
      path = configDir </> fileName
  exists <- doesFileExist path
  unless exists $ do
    putStrLn "Model has not been downloaded yet, downloading..."
    cloned <- doesDirectoryExist $ configDir </> ".git"
    let cmd   = "git clone git@gitlab.doc.ic.ac.uk:g206002126/grenade-assets.git " ++ configDir
        cmd'  = "git pull origin master"
        cmd'' = if cloned then cmd' else cmd
    putStrLn cmd''
    (_, Just hout, Just herr, _) <- createProcess (shell cmd'') { 
      cwd = Just configDir, std_out = CreatePipe, std_err = CreatePipe
    }
    hGetContents hout >>= putStrLn
    hGetContents herr >>= putStrLn
  return path

loadSerializedNetwork :: forall network. (Serialize network) => FilePath -> IO network
loadSerializedNetwork path = do
  modelData <- BS.readFile path
  either fail return $ runGet (get :: (Serialize network) => Get network) modelData
