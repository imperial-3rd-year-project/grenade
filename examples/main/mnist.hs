{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ExistentialQuantification #-}

import           Control.Applicative
import           Control.DeepSeq
import           Control.Monad
import           Control.Monad.Random
import           Control.Monad.Trans.Except

import           Data.Serialize

import qualified Data.ByteString              as B
import qualified Data.Attoparsec.Text         as A
import           Data.List                    (foldl')
import qualified Data.Text                    as T
import qualified Data.Text.IO                 as T
import qualified Data.Vector.Storable         as V
import           Data.Singletons.Prelude


import           Numeric.LinearAlgebra        (maxIndex, (<.>))
import qualified Numeric.LinearAlgebra.Static as SA
import           Numeric.LinearAlgebra.Data   (konst, size)

import           Options.Applicative

import           System.ProgressBar

import           Grenade
import           Grenade.Core.TrainingTypes
import           Grenade.Core.Training
import           Grenade.Utils.OneHot

import GHC.TypeLits
--
-- Note: Input files can be downloaded at https://www.kaggle.com/scolianni/mnistasjpg
--

-- The definition of our convolutional neural network.
-- In the type signature, we have a type level list of shapes which are passed between the layers.
-- One can see that the images we are inputing are two dimensional with 28 * 28 pixels.

-- It's important to keep the type signatures, as there's many layers which can "squeeze" into the gaps
-- between the shapes, so inference can't do it all for us.

-- With the mnist data from Kaggle normalised to doubles between 0 and 1, learning rate of 0.01 and 15 iterations,
-- this network should get down to about a 1.3% error rate.
--
--

type MNIST
  = Network
    '[ Convolution 1 10 5 5 1 1
     , Pooling 2 2 2 2
     , Relu
     , Convolution 10 16 5 5 1 1
     , Pooling 2 2 2 2
     , Reshape
     , Relu
     , FullyConnected 256 80
     , Logit
     , FullyConnected 80 10
     , Logit                    
     ]
    '[ 'D2 28 28
     , 'D3 24 24 10
     , 'D3 12 12 10
     , 'D3 12 12 10
     , 'D3 8 8 16
     , 'D3 4 4 16
     , 'D1 256
     , 'D1 256
     , 'D1 80
     , 'D1 80
     , 'D1 10
     , 'D1 10                   
     ]

data MnistOpts = MnistOpts FilePath          -- Training/Test data
                           Int               -- Iterations
                           Bool              -- Use Adam
                           (Optimizer 'SGD)  -- SGD settings
                           (Optimizer 'Adam) -- Adam settings
                           (Maybe FilePath)  -- Save path

mnist' :: Parser MnistOpts
mnist' = MnistOpts <$> argument str (metavar "TRAIN")
                   <*> option auto (long "iterations" <> short 'i' <> value 15)
                 <*> flag False True (long "use-adam" <> short 'a')
                 <*> (OptSGD
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
                       )
                 <*> (OptAdam
                       <$> option auto (long "alpha" <> short 'r' <> value 0.001)
                       <*> option auto (long "beta1" <> value 0.9)
                       <*> option auto (long "beta2" <> value 0.999)
                       <*> option auto (long "epsilon" <> value 1e-4)
                       <*> option auto (long "lambda" <> value 1e-3)
                      )
                 <*> optional (strOption (long "save"))

runFit :: FilePath -> Int -> Bool -> Optimizer opt1 -> Optimizer opt2 -> ExceptT String IO MNIST
runFit mnistPath iter useAdam sgd adam = do 
    lift $ putStrLn "Reading data..."
    allData <- readMNIST mnistPath
    let (trainData, validateData) = splitAt 33000 allData

    lift $ putStrLn "Training convolutional neural network..."
    lift $ fit trainData validateData options iter
  where
    options = if useAdam 
      then TrainingOptions 
        { optimizer = adam
        , verbose   = Full 
        , metrics   = []
        }
      else TrainingOptions 
        { optimizer = sgd
        , verbose   = Full 
        , metrics   = []
        }

main :: IO ()
main = do
    MnistOpts mnistPath iter useAdam sgd adam savePathM <- execParser (info (mnist' <**> helper) idm)
    res <- (runExceptT $ runFit mnistPath iter useAdam sgd adam) :: IO (Either String MNIST)
    case res of
      Right _  -> putStrLn "Success"
      Left err -> putStrLn err

readMNIST :: FilePath -> ExceptT String IO [(S ('D2 28 28), S ('D1 10))]
readMNIST mnist = ExceptT $ do
  mnistdata <- T.readFile mnist
  return $ traverse (A.parseOnly parseMNIST) (tail $ T.lines mnistdata)

parseMNIST :: A.Parser (S ('D2 28 28), S ('D1 10))
parseMNIST = do
  Just lab <- oneHot <$> A.decimal
  pixels   <- many (A.char ',' >> A.double)
  image    <- maybe (fail "Parsed row was of an incorrect size") pure (fromStorable . V.fromList $ map realToFrac ((/ 255) <$> pixels))
  return (image, lab)

saveNet :: MNIST -> FilePath -> IO ()
saveNet net path = B.writeFile path $ runPut (put net)