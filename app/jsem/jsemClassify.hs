{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}

--module JSeMclassify where

import System.FilePath ((</>),(<.>))
--import Control.Exception as E
import qualified Data.Text as T    --text
import qualified Data.Text.IO as T --text
import qualified Data.Aeson as A   --aeson

import qualified DyNet.Core as D      --dynet-haskell
import DyNet.Tools.Juman as Juman     --dynet-tools
import DyNet.Simple as D              --dynet-tools
import DyNet.Simple.LSTMclassifier    --dynet-tools

-- | JSeM classifier
--   Usage : stack exec jsem-run 1-128-10-256-412-88-20-100 test.txt
--               1-128-10-256-412-88-20-100: hyperparameters
--               test.txt: a text file of an inference data (the last line of which is the hypothesis)
main :: IO()
main = do
  let homedir = "/home/bekki/program/dynet-tools"
      loaddir = homedir </> "app" </> "jsem" </> "jsem_models"
--  E.catch (do
  -- | 初期化。第一引数はjson形式ファイル、第二引数は前提文・帰結文が書かれたテキストファイル。
  (aesonFileName:premisehypoFileName:_) <- D.setup
  -- | jsonファルからハイパーパラメータを読み出す。
  maybeHyperParams <- (A.decodeFileStrict $ loaddir </> aesonFileName <.> "json")
  hyperParams <- case maybeHyperParams of
        Nothing -> ioError $ userError $ aesonFileName ++ " not well-formed."
        Just json -> return json
  -- | テキストファイルから
  (premiseTx,hypoTx) <- (\lns -> (T.concat (init lns), last lns)) <$> T.lines <$> T.readFile premisehypoFileName
  premise <- Juman.processTextByJumanOnline premiseTx
  hypo <- Juman.processTextByJumanOnline hypoTx
  model <- D.createModel
  params <- setLSTMparams model hyperParams
  D.loadModel model $ loaddir </> (showHyperParams hyperParams) <.> "model"
  D.classify (buildClassifier params) (labels params) (premise,hypo) >>= T.putStrLn
--    ) $ \(E.SomeException e) -> print e

