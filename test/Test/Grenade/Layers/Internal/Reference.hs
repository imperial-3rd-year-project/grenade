{-# LANGUAGE RankNTypes #-}

module Test.Grenade.Layers.Internal.Reference where

import           Grenade.Types
import           Numeric.LinearAlgebra
import qualified Numeric.LinearAlgebra.Static as H
import           GHC.TypeLits

im2col :: Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2col nrows ncols srows scols m =
  let starts = fittingStarts (rows m) nrows srows (cols m) ncols scols
  in  im2colFit starts nrows ncols m

vid2col :: Int -> Int -> Int -> Int -> Int -> Int -> [Matrix RealNum] -> Matrix RealNum
vid2col nrows ncols srows scols inputrows inputcols ms =
  let starts = fittingStarts inputrows nrows srows inputcols ncols scols
      subs   = fmap (im2colFit starts nrows ncols) ms
  in  foldl1 (|||) subs

im2colFit :: [(Int,Int)] -> Int -> Int -> Matrix RealNum -> Matrix RealNum
im2colFit starts nrows ncols m =
  let imRows = fmap (\start -> flatten $ subMatrix start (nrows, ncols) m) starts
  in  fromRows imRows

col2vid :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> [Matrix RealNum]
col2vid nrows ncols srows scols drows dcols m =
  let starts = fittingStart (cols m) (nrows * ncols) (nrows * ncols)
      r      = rows m
      mats   = fmap (\s -> subMatrix (0,s) (r, nrows * ncols) m) starts
      colSts = fittingStarts drows nrows srows dcols ncols scols
  in  fmap (col2imfit colSts nrows ncols drows dcols) mats

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
col2im krows kcols srows scols drows dcols m =
  let rs       = map toList $ toColumns m 
      rs'      = zip [0..] rs
      indicies = (\[a,b] -> (a,b)) <$> sequence [[0..(krows-1)], [0..(kcols-1)]]
      accums   = concatMap (\(offset, column) -> zipWith (comb offset) indicies column) rs'
  in accum (konst 0 (drows, dcols)) (+) accums
  where     
    comb o (i, j) x = let w = (div (dcols - kcols) scols) + 1
                          (a, b) = divMod o w
                      in  ((i + srows * a, j + scols * b), x)

col2imfit :: [(Int,Int)] -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
col2imfit starts krows kcols drows dcols m =
  let indicies   = (\[a,b] -> (a,b)) <$> sequence [[0..(krows-1)], [0..(kcols-1)]]
      convs      = fmap (zip indicies . toList) . toRows $ m
      pairs      = zip convs starts
      accums     = concatMap (\(conv',(stx',sty')) -> fmap (\((ix,iy), val) -> ((ix + stx', iy + sty'), val)) conv') pairs
  in  accum (konst 0 (drows, dcols)) (+) accums

poolForward :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
poolForward nrows ncols srows scols outputRows outputCols m =
  let starts = fittingStarts (rows m) nrows srows (cols m) ncols scols
  in  poolForwardFit starts nrows ncols outputRows outputCols m

poolForwardList :: Functor f => Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> f (Matrix RealNum) -> f (Matrix RealNum)
poolForwardList nrows ncols srows scols inRows inCols outputRows outputCols ms =
  let starts = fittingStarts inRows nrows srows inCols ncols scols
  in  poolForwardFit starts nrows ncols outputRows outputCols <$> ms

poolForwardFit :: [(Int,Int)] -> Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum
poolForwardFit starts nrows ncols _ outputCols m =
  let els    = fmap (\start -> maxElement $ subMatrix start (nrows, ncols) m) starts
  in  matrix outputCols els

poolBackward :: Int -> Int -> Int -> Int -> Matrix RealNum -> Matrix RealNum -> Matrix RealNum
poolBackward krows kcols srows scols inputMatrix gradientMatrix =
  let inRows     = rows inputMatrix
      inCols     = cols inputMatrix
      starts     = fittingStarts inRows krows srows inCols kcols scols
  in  poolBackwardFit starts krows kcols inputMatrix gradientMatrix

poolBackwardList :: Functor f => Int -> Int -> Int -> Int -> Int -> Int -> f (Matrix RealNum, Matrix RealNum) -> f (Matrix RealNum)
poolBackwardList krows kcols srows scols inRows inCols inputMatrices =
  let starts     = fittingStarts inRows krows srows inCols kcols scols
  in  uncurry (poolBackwardFit starts krows kcols) <$> inputMatrices

poolBackwardFit :: [(Int,Int)] -> Int -> Int -> Matrix RealNum -> Matrix RealNum -> Matrix RealNum
poolBackwardFit starts krows kcols inputMatrix gradientMatrix =
  let inRows     = rows inputMatrix
      inCols     = cols inputMatrix
      inds       = fmap (\start -> maxIndex $ subMatrix start (krows, kcols) inputMatrix) starts
      grads      = toList $ flatten gradientMatrix
      grads'     = zip3 starts grads inds
      accums     = fmap (\((stx',sty'),grad,(inx, iny)) -> ((stx' + inx, sty' + iny), grad)) grads'
  in  accum (konst 0 (inRows, inCols)) (+) accums

-- | These functions are not even remotely safe, but it's only called from the statically typed
--   commands, so we should be good ?!?!?
--   Returns the starting sub matrix locations which fit inside the larger matrix for the
--   convolution. Takes into account the stride and kernel size.
fittingStarts :: Int -> Int -> Int -> Int -> Int -> Int -> [(Int,Int)]
fittingStarts nrows kernelrows steprows ncols kernelcols stepcolsh =
  let rs = fittingStart nrows kernelrows steprows
      cs = fittingStart ncols kernelcols stepcolsh
      ls = sequence [rs, cs]
  in  fmap (\[a,b] -> (a,b)) ls

-- | Returns the starting sub vector which fit inside the larger vector for the
--   convolution. Takes into account the stride and kernel size.
fittingStart :: Int -> Int -> Int -> [Int]
fittingStart width kernel steps =
  let go left | left + kernel < width
              = left : go (left + steps)
              | left + kernel == width
              = [left]
              | otherwise
              = []
  in go 0

convBackProp :: Matrix RealNum -> Int -> Int -> Int 
             -> Matrix RealNum -> Int -> Int -> Int 
             -> Matrix RealNum -> Int -> Int
             -> Int -> Int 
             -> (Matrix RealNum, Matrix RealNum)
convBackProp input channels inRows inCols kernel filters kernelRows kernelCols dout outRows outCols strideRows strideCols =
  let fs       = [0..filters-1]
      hs       = [let h_start = h * strideRows in (h, h_start) | h <- [0..outRows - 1]]
      ws       = [let w_start = w * strideCols in (w, w_start) | w <- [0..outCols - 1]]
      fhws     = [(f, h, w) | f <- fs, h <- hs, w <- ws]
      dX_accum = concatMap dxLoop fhws
      dX       = accum (konst 0 (channels * inRows, inCols)) (+) dX_accum
      dW_accum = concatMap dwLoop fhws
      dW       = accum (konst 0 (kernelRows * kernelCols * channels, filters)) (+) dW_accum
  in (dX, dW)
  where 
    dxLoop (f, (h, h_start), (w, w_start)) 
      = [ ((c * inRows + h_start + i, w_start + j), (indexAtdOut f h w) * (indexAtKernel f c i j)) | i <- [0..kernelRows-1], j <- [0..kernelCols-1], c <- [0..channels-1]]
    
    dwLoop (f, (h, h_start), (w, w_start))
      = [ ((c * kernelRows * kernelCols + i * kernelCols + j, f), (indexAtdOut f h w) * (indexAtIn c (h_start + i) (w_start + j))) | i <- [0..kernelRows-1], j <- [0..kernelCols-1], c <- [0..channels-1]]

    indexAtIn       c x y = input `atIndex` (c * inRows + x, y)
    indexAtdOut     c x y = dout `atIndex` (c * outRows + x, y)
    indexAtKernel f c x y = kernel `atIndex` (c * kernelRows * kernelCols + x * kernelCols + y, f)

convForwards :: Matrix RealNum -> Int -> Int -> Int 
             -> Matrix RealNum -> Int -> Int -> Int 
             -> Int -> Int
             -> Int -> Int 
             -> Matrix RealNum
convForwards input channels inRows inCols kernel filters kernelRows kernelCols outRows outCols strideRows strideCols =
  let fs     = [0..filters-1]
      hs     = [(h, h * strideRows) | h <- [0..outRows - 1], h * strideRows + kernelRows - 1 < inRows]
      ws     = [(w, w * strideCols) | w <- [0..outCols - 1], w * strideCols + kernelCols - 1 < inCols]
      fhws   = [(f, h, w) | f <- fs, h <- hs, w <- ws]
      accums = map loopIter fhws
  in accum (konst 0 (filters * outRows, outCols)) (+) accums
  where 
    loopIter (f, (h, h_start), (w, w_start)) 
      = ((f * outRows + h, w), sum [ (indexAtKernel f c i j) * (indexAtIn c (h_start + i) (w_start + j)) | i <- [0..kernelRows-1], j <- [0..kernelCols-1], c <- [0..channels-1]])
    
    indexAtIn c x y = input `atIndex` (c * inRows + x, y)
    indexAtKernel f c x y = kernel `atIndex` (c * kernelRows * kernelCols + x * kernelCols + y, f)

convForwardsWithPadding :: Matrix RealNum -> Int -> Int -> Int 
                        -> Matrix RealNum -> Int -> Int -> Int 
                        -> Int -> Int
                        -> Int -> Int
                        -> Int -> Int -> Int -> Int  
                        -> Matrix RealNum
convForwardsWithPadding input channels inRows inCols kernel filters kernelRows kernelCols outRows outCols strideRows strideCols padl padt padr padb =
  let accums      = [((c * padded_r + x + padt, y + padl), input `atIndex` (c * inRows + x, y)) | x <- [0..inRows-1], y <- [0..inCols-1], c <- [0..channels-1]]
      padded_r    = inRows + padt + padb
      padded_c    = inCols + padl + padr
      paddedInput = accum (konst 0 (channels * padded_r, padded_c)) (+) accums
  in convForwards paddedInput channels padded_r padded_c kernel filters kernelRows kernelCols outRows outCols strideRows strideCols 

convBackPropWithPadding :: Matrix RealNum -> Int -> Int -> Int 
                        -> Matrix RealNum -> Int -> Int -> Int 
                        -> Matrix RealNum -> Int -> Int
                        -> Int -> Int 
                        -> Int -> Int -> Int -> Int  
                        -> (Matrix RealNum, Matrix RealNum)
convBackPropWithPadding input channels inRows inCols kernel filters kernelRows kernelCols dout outRows outCols strideRows strideCols padl padt padr padb =
  let accums      = [((c * padded_r + x + padt, y + padl), input `atIndex` (c * inRows + x, y)) | x <- [0..inRows-1], y <- [0..inCols-1], c <- [0..channels-1]]
      padded_r    = inRows + padt + padb
      padded_c    = inCols + padl + padr
      paddedInput = accum (konst 0 (channels * padded_r, padded_c)) (+) accums
      (dX', dW)   = convBackProp paddedInput channels padded_r padded_c kernel filters kernelRows kernelCols dout outRows outCols strideRows strideCols 
      accums'     = [((c * inRows + x, y), dX' `atIndex` (c * padded_r + x + padt, y + padl)) | x <- [0..inRows-1], y <- [0..inCols-1], c <- [0..channels-1]]
      dX          = accum (konst 0 (channels * inRows, inCols)) (+) accums'
  in  (dX, dW)

biasConvForwards :: Matrix RealNum -> Int -> Int -> Int 
                 -> Matrix RealNum -> Int -> Int -> Int 
                 -> Vector RealNum
                 -> Int -> Int
                 -> Int -> Int 
                 -> Matrix RealNum
biasConvForwards input channels inRows inCols kernel filters kernelRows kernelCols bias outRows outCols strideRows strideCols =
  let fs     = [0..filters-1]
      hs     = [(h, h * strideRows) | h <- [0..outRows - 1], h * strideRows + kernelRows - 1 < inRows]
      ws     = [(w, w * strideCols) | w <- [0..outCols - 1], w * strideCols + kernelCols - 1 < inCols]
      fhws   = [(f, h, w) | f <- fs, h <- hs, w <- ws]
      accums = map loopIter fhws
  in accum (konst 0 (filters * outRows, outCols)) (+) accums
  where 
    loopIter (f, (h, h_start), (w, w_start)) 
      = ((f * outRows + h, w), indexAtBias f + sum [ (indexAtKernel f c i j) * (indexAtIn c (h_start + i) (w_start + j)) | i <- [0..kernelRows-1], j <- [0..kernelCols-1], c <- [0..channels-1]])
    
    indexAtBias i = bias `atIndex` i
    indexAtIn c x y = input `atIndex` (c * inRows + x, y)
    indexAtKernel f c x y = kernel `atIndex` (c * kernelRows * kernelCols + x * kernelCols + y, f)

biasConvForwardsWithPadding :: Matrix RealNum -> Int -> Int -> Int 
                        -> Matrix RealNum -> Int -> Int -> Int 
                        -> Vector RealNum
                        -> Int -> Int
                        -> Int -> Int
                        -> Int -> Int -> Int -> Int  
                        -> Matrix RealNum
biasConvForwardsWithPadding input channels inRows inCols kernel filters kernelRows kernelCols bias outRows outCols strideRows strideCols padl padt padr padb =
  let accums      = [((c * padded_r + x + padt, y + padl), input `atIndex` (c * inRows + x, y)) | x <- [0..inRows-1], y <- [0..inCols-1], c <- [0..channels-1]]
      padded_r    = inRows + padt + padb
      padded_c    = inCols + padl + padr
      paddedInput = accum (konst 0 (channels * padded_r, padded_c)) (+) accums
  in biasConvForwards paddedInput channels padded_r padded_c kernel filters kernelRows kernelCols bias outRows outCols strideRows strideCols 

biasConvBackProp :: Matrix RealNum -> Int -> Int -> Int 
                 -> Matrix RealNum -> Int -> Int -> Int 
                 -> Matrix RealNum -> Int -> Int
                 -> Int -> Int 
                 -> (Matrix RealNum, Matrix RealNum, Vector RealNum)
biasConvBackProp input channels inRows inCols kernel filters kernelRows kernelCols dout outRows outCols strideRows strideCols =
  let (dX, dW)  = convBackProp input channels inRows inCols kernel filters kernelRows kernelCols dout outRows outCols strideRows strideCols
      db_accums = [ (f, sum [ indexAtdOut f i j | i <- [0..outRows-1], j <- [0..outCols-1]]) | f <- [0..filters-1]]
      db        = accum (konst 0 filters) (+) db_accums
  in (dX, dW, db)
  where 
    indexAtdOut     c x y = dout `atIndex` (c * outRows + x, y)

biasConvBackPropWithPadding :: Matrix RealNum -> Int -> Int -> Int 
                            -> Matrix RealNum -> Int -> Int -> Int 
                            -> Matrix RealNum -> Int -> Int
                            -> Int -> Int 
                            -> Int -> Int -> Int -> Int  
                            -> (Matrix RealNum, Matrix RealNum, Vector RealNum)
biasConvBackPropWithPadding input channels inRows inCols kernel filters kernelRows kernelCols dout outRows outCols strideRows strideCols padl padt padr padb =
  let accums        = [((c * padded_r + x + padt, y + padl), input `atIndex` (c * inRows + x, y)) | x <- [0..inRows-1], y <- [0..inCols-1], c <- [0..channels-1]]
      padded_r      = inRows + padt + padb
      padded_c      = inCols + padl + padr
      paddedInput   = accum (konst 0 (channels * padded_r, padded_c)) (+) accums
      (dX', dW, db) = biasConvBackProp paddedInput channels padded_r padded_c kernel filters kernelRows kernelCols dout outRows outCols strideRows strideCols 
      accums'       = [((c * inRows + x, y), dX' `atIndex` (c * padded_r + x + padt, y + padl)) | x <- [0..inRows-1], y <- [0..inCols-1], c <- [0..channels-1]]
      dX            = accum (konst 0 (channels * inRows, inCols)) (+) accums'
  in  (dX, dW, db)

naiveFullyConnectedRunForwards :: forall i o. (KnownNat i, KnownNat o) 
                               => H.L o i       -- Weights
                               -> H.R o         -- Biases
                               -> H.R i         -- Input
                               -> (H.R i, H.R o)  -- (Tape, Output)
naiveFullyConnectedRunForwards w b i = (i, b + (w H.#> i))

naiveFullyConnectedBackprop :: forall i o. (KnownNat i, KnownNat o) 
                            => H.L o i             -- Weights
                            -> (H.R i, H.R o)        -- (Tape, Output)
                            -> (H.L o i, H.R o, H.R i) -- (NablaW, NablaB, derivatives)
naiveFullyConnectedBackprop w (tape, out) = (w', b', d)
  where
    b' = out
    w' = H.outer out tape
    d  = (H.tr w) H.#> out

-- Implementation reference: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
naiveLRNForwards :: RealNum -> RealNum -> RealNum -> Int -> [[[RealNum]]] -> [[[RealNum]]]
naiveLRNForwards a b k n values = [
    [[ g ch ro co | co <- [0..length (values!!ch!!ro) - 1] ] | ro <- [0..length (values!!ch) - 1]] | ch <- [0..cs-1]
  ]
  where
    cs = length values
    f ch ro co = values!!ch!!ro!!co
    g ch ro co = (f ch ro co) / den
      where
        den  = den' ** b
        sub = floor ((fromIntegral n) / 2     :: RealNum)
        add = floor ((fromIntegral n - 1) / 2 :: RealNum)
        lower = maximum [0, ch - sub]
        upper = minimum [cs - 1, ch + add]
        summation = sum [ (f j ro co) ** 2 | j <- [lower..upper]]
        den' = k + a * summation

naiveLRNBackwards :: RealNum         -- a
                    -> RealNum       -- b
                    -> RealNum       -- k
                    -> Int          -- n
                    -> [[[RealNum]]] -- inputs
                    -> [[[RealNum]]] -- backpropagated error
                    -> [[[RealNum]]] -- error to propagate further
naiveLRNBackwards a b k n values errs = [
    [[ ng ch ro co | co <- [0..length (values!!ch!!ro) - 1] ] | ro <- [0..length (values!!ch) - 1]] | ch <- [0..cs-1]
  ]
  where
    cs = length values
    f  ch ro co = values!!ch!!ro!!co
    nf ch ro co = errs!!ch!!ro!!co
    c ch ro co = den
      where
        sub = floor ((fromIntegral n) / 2     :: RealNum)
        add = floor ((fromIntegral n - 1) / 2 :: RealNum)
        lower = maximum [0, ch - sub]
        upper = minimum [cs - 1, ch + add]
        summation = sum [ (f j ro co) ** 2 | j <- [lower..upper]]
        den = k + a * summation
    ng ch ro co = t1 - t2
      where
        t1 = (c ch ro co) ** (-b) * (nf ch ro co)
        t2 = 2 * b * a * (f ch ro co) * (c ch ro co) ** (-b - 1) * s
        s  = sum [(f q ro co) * (nf q ro co) | q <- [lower..upper]]

        sub = floor ((fromIntegral n) / 2     :: RealNum)
        add = floor ((fromIntegral n - 1) / 2 :: RealNum)
        lower = maximum [0, ch - sub]
        upper = minimum [cs - 1, ch + add]
