using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace WindowsFormsApp1
{
    class SentimentIssue
    {
        public SentimentIssue(string feature)
        {
            Text = feature;
        }

        [LoadColumn(1)]
        public string Text { get; set; }
        [LoadColumn(0)]
        public bool Label { get; set; }
    }
}
