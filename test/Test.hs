{-# OPTIONS -Wall #-}
{-# LANGUAGE OverloadedStrings #-}

import qualified System.Environment as E --base
import qualified DyNet.Core as D     --dynet-haskell

main :: IO ()
main = do
  argv <- E.getArgs
  _ <- D.initialize' argv
  model <- D.createModel
  _ <- D.addParameters' model [10::Int, 10::Int]
  putStrLn "ok"
  
