{-# OPTIONS -Wall #-}
{-# LANGUAGE OverloadedStrings #-}

module DyNet.Tools.Chart (
  drawLearningCurve
  ) where

import Data.List (transpose)
import Graphics.Gnuplot.Simple
import DyNet.Simple (LearningChart)

-- | LearningChartは各行が
-- (0, [107.26212, 112.046524 94.8856 91.56159 85.57186 25.32702])
-- (1, [58.097065, 114.710655 101.360596 98.431046 84.57611 24.729351])
-- なので、これから
-- (sty "batch-1", [(0,107.26212),(1,58.09765),...])
-- を剥ぎとって（どういう意味？？
-- (0, [112.046524 94.8856 91.56159 85.57186 25.32702])
-- (1, [114.710655 101.360596 98.431046 84.57611 24.729351])
-- を返して再帰する。

testData :: LearningChart
testData = [
  (0, [107.26212, 112.046524, 94.8856,    91.56159,  85.57186,  25.32702]),
  (1, [58.097065, 114.710655, 101.360596, 98.431046, 84.57611,  24.729351]),
  (2, [61.99805,  89.96048,   86.6363,    89.605865, 88.304985, 23.252378]),
  (3, [62.932064, 91.324265,  83.364265,  85.87307,  82.94884,  20.744677]),
  (4, [48.18716,  97.32671,   84.417404,  85.4241,   79.75656,  17.75763]),
  (5, [46.595695, 83.82408,   73.23103,   79.69995,  80.71408,  15.31874]),
  (6, [47.275078, 79.2172,    67.13917,   73.52796,  74.99438,  11.9399395]),
  (7, [38.305492, 82.1474,    64.91645,   70.82271,  74.180565, 8.756168]),
  (8, [32.533005, 73.59252,   51.76041,   63.517025, 76.30533,  6.813904]),
  (9, [29.234098, 68.950035,  43.700676,  59.53903,  70.3834,   4.861396]),
  (10,[21.261477, 75.850494,  42.84322,   58.941006, 65.03832,  3.999124])
  ]

drawLearningCurve :: FilePath -> String -> LearningChart -> IO()
drawLearningCurve filepath title learningchart = do
  let maxepoch  = fromIntegral $ length learningchart
      dat       = transpose $ map (\(epoch,losses) -> map (\loss -> (fromIntegral epoch, (realToFrac loss)::Double)) losses) learningchart
      styleddat = zip (map (\epoch -> let linetitle = "batch-" ++ (show epoch)
                                      in PlotStyle LinesPoints (CustomStyle [LineTitle linetitle])
                                      ) [(0::Int)..]) dat -- | [(PlotStyle, [(Double,Double)])]
      graphstyle = [(PNG filepath),(Title title),(XLabel "epoch"),(YLabel "loss"),(XRange (0,maxepoch))]
  plotPathsStyle graphstyle styleddat

