using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace MachineLearningTesting
{
    class TestModel
    {
        static void Main(string[] args)
        {
            string trainFile = "../../../Short.txt";
            string testFile = "../../../Long.txt";
            string saveFile = "../../../MLMODEL.zip";
            Model m = new Model(trainFile, testFile, saveFile);

            m.Train();
            m.Test();
            
            Console.ReadLine();
        }
    }
}
