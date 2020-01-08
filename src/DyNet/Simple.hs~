{-# OPTIONS -Wall #-}
{-# LANGUAGE OverloadedStrings, TypeSynonymInstances, FlexibleInstances #-}

module DyNet.Simple (
  Epoch,
  Loss,
  LearningChart,
  setup,
  kfold,
  train,
  classify,
  evaluate,
  saveModel,
  loadModel,
  gridsearch
  ) where

import System.Environment (getArgs)         --base
import Data.List (nub,sortOn)               --base
import Control.Monad (forM,forM_,guard)     --base
import Data.Hashable (Hashable)             --hashable
import qualified Data.Text    as StrictT    --text
import qualified Data.Text.IO as StrictT    --text
import qualified DyNet.Core as D    --dynet-haskell
import qualified DyNet.Expr as D    --dynet-haskell
import qualified DyNet.Dict as D    --dynet-haskell
import qualified DyNet.Train as D   --dynet-haskell
import qualified DyNet.IO as D      --dynet-haskell
import qualified DyNet.Vector as V  --dynet-haskell

type Label = StrictT.Text
type Epoch = Int
type Loss = Float
type LearningChart = [(Epoch,[Loss])]

-- | 初期化・メモリの確保。返り値は引数のリスト。
setup :: IO [String]
setup = do
  argv <- getArgs
  D.initialize' argv

-- | datをi分割する。
kfold :: Int -> [a] -> ([a],[a])
kfold i dat = foldr (\(j,d) (trainData,testData) -> if j `mod` i == i-1
                                                             then (trainData,d:testData)
                                                             else (d:trainData,testData)
                                                             ) ([],[]) $ zip [0..] dat

miniBatch :: Int -> [a] -> [[a]]
miniBatch _    [] = []
miniBatch size xs =
  let (x, xs') = splitAt size xs in
  x:miniBatch size xs'

-- | 
train :: (D.Trainer t, Hashable k, Eq k, Show k)
           => Int  -- ^ number of iteration
           -> Int  -- ^ size of minibatches
           -> [(a,k)] -- ^ 訓練データ
           -> (D.ComputationGraph -> a -> IO D.Expression) -- ^ 計算グラフ
           -> D.Dict k -- ^ 出力ラベルから作ったD.Dict
           -> t　　　 -- ^ 訓練器
           -> IO(LearningChart)
train iter batchSize trainData classifier outputDict trainer = do
  forM [0..iter] $ \epoch -> do
    putStr $ "epoch " ++ (show epoch) ++ ":"
    losses <- forM (zip [(0::Int)..] $ miniBatch batchSize trainData) $ \(jth,oneBatch) -> do
      putStr $ "\t [" ++ (show jth) ++ "] " 
      D.withNewComputationGraph $ \cg -> do
        lossExprs <- forM oneBatch $ \(input,output) -> do
                                expr <- classifier cg input
                                outputID <- D.fromString outputDict output
                                D.pickneglogsoftmax expr outputID
        lossExpr <- D.sum lossExprs
        loss <- D.forward cg lossExpr >>= D.asScalar
        D.backward cg lossExpr
        D.update trainer
        putStr $ "loss=" ++ (show loss)
        return loss
    D.status trainer
    putStr "\n"
    --D.updateEpoch trainer 1.0
    return (epoch, losses)

classify :: (D.ComputationGraph -> a -> IO D.Expression) -> D.Dict(Label) -> a -> IO(Label)
classify classifier outputDict input = do
  D.withNewComputationGraph $ \cg -> do
    expr <- classifier cg input
    v <- D.forward cg expr >>= D.asVector >>= V.toList
    D.fromIndex outputDict (D.argmax v)

evaluate :: [(a,Label)] -> (D.ComputationGraph -> a -> IO D.Expression) -> D.Dict(Label) -> [Label] -> IO(Float)
evaluate testData classifier outputDict outputLabels = do
  let (testInputs,testOutputs) = unzip testData
  putStr "Calculating predictions: "
  predictions <- forM testInputs $ \input -> do
                                             putStr "o"
                                             classify classifier outputDict input
  putStr "\n"
  let results = zip predictions testOutputs
      overallAccuracy = calcOverallAccuracy results
      averageAccuracy = calcAverageAccuracy results
  putStrLn $ "Overall accuracy = " ++ (show overallAccuracy)
  putStrLn $ "Average accuracy = " ++ (show averageAccuracy)
  printConfusionMatrix outputLabels $ confusionMatrix outputLabels results
  return overallAccuracy

calcOverallAccuracy :: [(Label,Label)] -> Float
calcOverallAccuracy results = 
  let denominator = fromIntegral $ length results
      numerator = fromIntegral $ length $ [(x,y) | (x,y) <- results, x == y] 
  in (numerator / denominator)

calcAverageAccuracy :: [(Label,Label)] -> Float
calcAverageAccuracy results =
  let labels = nub $ snd $ unzip results
      denominator = fromIntegral $ length labels
      numerator = sum $ map (\label -> calcOverallAccuracy [(x,y) | (x,y) <- results, y == label]) labels
  in (numerator / denominator)

confusionMatrix :: [Label] -> [(Label,Label)] -> [[Int]]
confusionMatrix outputLabels results = do
  prediction <- outputLabels
  return $ do
           answer <- outputLabels
           return $ length $ do
                             (x,y) <- results
                             guard $ x == prediction
                             guard $ y == answer
                             return (x,y)

printConfusionMatrix :: [Label] -> [[Int]] -> IO()
printConfusionMatrix outputLabels matrix = do
  StrictT.putStr "\t|\t" 
  StrictT.putStrLn $ StrictT.intercalate "\t|\t" outputLabels
  putStr "-----"
  forM_ outputLabels (\_ -> putStr "-------------------")
  putStr "\n"
  mapM_ (\(label,ints) -> do
                          StrictT.putStr label
                          StrictT.putStr "\t|\t"
                          StrictT.putStrLn $ StrictT.intercalate "\t|\t" $ map (StrictT.pack . show) ints) $ zip outputLabels matrix

saveModel :: D.Model -> FilePath -> IO()
saveModel model filepath = D.createSaver' filepath >>= flip D.saveModel' model
                           
loadModel :: D.Model -> FilePath -> IO()
loadModel model filepath = D.createLoader filepath >>= flip D.populateModel' model

gridsearch :: [IO(D.Model,String,Float)] -> IO()
gridsearch commands = do
  results <- sequence commands
  let ranking = reverse $ sortOn (\(_,_,a) -> a) results
  --case head ranking of
  --  (model, setting, _) -> saveModel model $ "model" </> setting ++ ".model"
  forM_ ranking $ \(_, setting, accuracy) ->
                    putStrLn $ "Accuracy: " ++ (show accuracy) ++ "  Setting: " ++ setting

--writeLearningChart :: FilePath -> LearningChart -> IO()
--writeLearningChart filepath learningchart =
--  writeFile filepath $ intercalate "\n" $ map (\(epoch,losses) -> (show epoch) ++ " " ++ (intercalate " " $ map show losses)) learningchart

