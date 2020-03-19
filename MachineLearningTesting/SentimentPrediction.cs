﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearningTesting
{
    class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probablity { get; set; }
        public float Score { get; set; }
    }
}
