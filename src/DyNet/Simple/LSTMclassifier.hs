{-# OPTIONS -Wall #-}
{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}

module DyNet.Simple.LSTMclassifier (
  jsemlabels,
  Token,
  Label,
  LSTMparams(..),
  jsemlabels,
  HyperParams(..),
  showHyperParams,
  setLSTMparams,
  buildClassifier,
  buildLSTM
  ) where

import Prelude hiding (words)
import GHC.Generics
import Data.List (intercalate)        --base
import Control.Monad (forM)           --base
import qualified Data.Text as StrictT --text
import qualified Data.Aeson as A      --aeson
import qualified DyNet.Core as D      --dynet-haskell
import qualified DyNet.Expr as D      --dynet-haskell
import qualified DyNet.RNN as D       --dynet-haskell
import qualified DyNet.Dict as D      --dynet-haskell
import qualified DyNet.Train as D     --dynet-haskell
import qualified DyNet.Tools.Juman as Juman --dynet-tools

type Token = StrictT.Text
type Label = StrictT.Text

data LSTMparams = LSTMparams {
  words :: D.Dict Token,
  poss :: D.Dict Token,
  labels :: D.Dict Label,
  wordEmbed :: D.LookupParameter,
  posEmbed :: D.LookupParameter,
  w1 :: D.Parameter,
  w2 :: D.Parameter,
  w3 :: D.Parameter,
  fwdRNN :: D.VanillaLSTMBuilder
  --bwdRNN :: D.VanillaLSTMBuilder
  } 

jsemlabels :: [Label]
jsemlabels = ["YES","NO","UNKNOWN"]

data HyperParams = HyperParams {
  numOfLSTMlayers :: Int, -- ^ layers ToDo: 層数？
  wordEmbedDim :: Int,    -- ^ wordEmbedDim
  posEmbedDim :: Int,     -- ^ posEmbedDim
  lstmDim :: Int,         -- ^ lstmDim
  hidden1Dim :: Int,      -- ^ hidden1Dim
  hidden2Dim :: Int,      -- ^ hidden2Dim
  iteration :: Int,       -- ^ number of iteration
  batchSize :: Int,       -- ^ number of minibatch size
  wordslist :: [Token],   -- ^ words
  posslist :: [Token],    -- ^ part-of-speeches
  labelslist :: [Label]   -- ^ answer labels
  } deriving (Generic, Show)

instance A.ToJSON HyperParams
instance A.FromJSON HyperParams

-- | HyperParamsを文字列化（ハイフンで繋ぐだけ）
showHyperParams :: HyperParams -> String
showHyperParams (HyperParams numOfLSTMlayers wordEmbedDim posEmbedDim lstmDim hidden1Dim hidden2Dim iteration batchSize _ _ _) =
  intercalate "-" $ map show [numOfLSTMlayers,wordEmbedDim,posEmbedDim,lstmDim,hidden1Dim,hidden2Dim,iteration,batchSize]

-- | 
setLSTMparams :: D.Model -> HyperParams -> IO(LSTMparams)
setLSTMparams model (HyperParams numOfLSTMlayers wordEmbedDim posEmbedDim lstmDim hidden1Dim hidden2Dim _ _ words poss labels) =
  LSTMparams <$> D.createDict words (Just "<unkw>") -- ToDo: D.createDictの第二引数は見つからない時に返す値か？
         <*> D.createDict poss (Just "<unkp>")
         <*> D.createDict labels Nothing
         <*> D.addLookupParameters' model (length words + 1) [wordEmbedDim]
         <*> D.addLookupParameters' model (length poss + 1) [posEmbedDim]
         <*> D.addParameters' model [hidden1Dim, lstmDim*3]     -- w1
         <*> D.addParameters' model [hidden2Dim, hidden1Dim]    -- w2
         <*> D.addParameters' model [length labels, hidden2Dim] -- w3
         <*> D.createVanillaLSTMBuilder numOfLSTMlayers (wordEmbedDim+posEmbedDim) lstmDim model False

type JumanWord = (Juman.Kihon,Juman.Hinsi)

buildClassifier :: LSTMparams -> D.ComputationGraph -> ([JumanWord],[JumanWord]) -> IO D.Expression
buildClassifier params cg (premise,hypo) = do
  _W1 <- D.parameter cg (w1 params)
  _W2 <- D.parameter cg (w2 params)
  _W3 <- D.parameter cg (w3 params)
  premiseVec <- last <$> buildLSTM params cg premise
  hypoVec <- last <$> buildLSTM params cg hypo
  mult <- D.cmult premiseVec hypoVec
  _W3 `D.mul` (D.selu (_W2 `D.mul` (D.selu (_W1 `D.mul` (D.concat' [premiseVec, hypoVec, mult])))))

buildLSTM :: LSTMparams -> D.ComputationGraph -> [JumanWord] -> IO [D.Expression]
buildLSTM params cg inputs = do
  D.newGraph' (fwdRNN params) cg -- ToDo: これ何してんだっけ. D.parameter
  wembs <- forM inputs $ \(word,pos) -> do
             wordVec <- D.fromString (words params) word >>= D.lookup cg (wordEmbed params)
             posVec <- D.fromString (poss params) pos >>= D.lookup cg (posEmbed params)
             return $ D.concat' [wordVec, posVec]
  D.startNewSequence' (fwdRNN params) -- ToDo: これは確か初期値ベクトルをセットしている
  mapM (D.addInput (fwdRNN params)) wembs


