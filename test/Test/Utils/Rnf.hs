module Test.Utils.Rnf where

import Control.Exception
import Control.DeepSeq
import Data.List

{-------------------------------------------
  Contains utility functions to test the
  correctness of NFData instances for layers.
--------------------------------------------}

noErr :: String
noErr = "No error"

expectedErrStr :: String
expectedErrStr = "An error has occured!"

rnfRaisedErr :: String -> Bool
rnfRaisedErr x = expectedErrStr `isPrefixOf` x

rnfNoError :: String -> Bool
rnfNoError x = noErr `isPrefixOf` x

tryEvalRnf :: (NFData a) => a -> IO String
tryEvalRnf x = catch ((evaluate $ rnf x) >> pure noErr) $ \e -> do
  let err = show (e :: ErrorCall)
  return err