using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using System.Drawing;

namespace WindowsFormsApp1
{
    public class Model
    {
        //Step 2: Create object of MLContext so that we can access the features of machine learning
        private MLContext mlContext { get; set; }
        private IDataView trainDataView { get; set; } //DataView for Data that we are using to train
        private IDataView testDataView { get; set; } //DataView for Data that we are using to test the ALREADY trained model
        public IEstimator<ITransformer> pipeline { get; set; }
        public ITransformer trainedModel  { get; set; }
        public string trainDataLocation { get; set; }
        public string testDataLocation { get; set; }
        public int testFraction { get; set; }
        public string saveDataLocation { get; set; }
        public CalibratedBinaryClassificationMetrics metrics{ get; set; }

        //trainDL: Location of Data model we will be using to train
        //testDL: Location of Data that we are using to test the ALREADY trained model
        public Model(string trainDL, string testDL, string sDL)
        {
            //Step 1: Create Context for testing and training
            mlContext = new MLContext();
            trainDataLocation = trainDL;
            testDataLocation = testDL;
            saveDataLocation = sDL;
            //Step 3: Convert input and output files into IDataView
            trainDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(trainDL, hasHeader: true);
            testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(testDL, hasHeader: true);
            //Step 4: Create pipeline, a
            #region pipelineExplanation 
            /** FROM https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
             * An Azure Machine Learning pipeline is an independently executable workflow 
             * of a complete machine learning task. Subtasks are encapsulated as a series 
             * of steps within the pipeline. An Azure Machine Learning pipeline can be as 
             * simple as one that calls a Python script, so may do just about anything.Pipelines 
             * should focus on machine learning tasks such as:
             * 
             * Data preparation including importing, validating and cleaning, munging and transformation, normalization, and staging
             * Training configuration including parameterizing arguments, filepaths, and logging / reporting configurations
             * Training and validating efficiently and repeatedly. Efficiency might come from specifying specific data subsets, 
             *      different hardware compute resources, distributed processing, and progress monitoring
             * Deployment, including versioning, scaling, provisioning, and access control
            ***/
            #endregion
            pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                inputColumnName: nameof(SentimentIssue.Text))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",
                featureColumnName: "Features"));
        }
        public Model(string dataLocation, string sDL, int testFraction)
        {
            mlContext = new MLContext();
            this.testFraction = testFraction;
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(dataLocation, hasHeader: true);
            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: Convert.ToDouble(testFraction)/100);
            trainDataView = trainTestSplit.TrainSet;
            testDataView = trainTestSplit.TestSet;
            saveDataLocation = sDL;
            pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                inputColumnName: nameof(SentimentIssue.Text))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",
                featureColumnName: "Features"));
        }
        #region Unused Model Constructors
        //public Model(string trainDL, string testDL)
        //{
        //    //Step 1: Create Context for testing and training
        //    mlContext = new MLContext();
        //    this.trainDataLocation = trainDL;
        //    this.testDataLocation = testDL;
        //    //Step 3: Convert input and output files into IDataView
        //    trainDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(trainDL, hasHeader: true);
        //    testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(testDL, hasHeader: true);
        //    //Step 4: Create pipeline, a
        //    #region pipelineExplanation 
        //    /** FROM https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
        //     * An Azure Machine Learning pipeline is an independently executable workflow 
        //     * of a complete machine learning task. Subtasks are encapsulated as a series 
        //     * of steps within the pipeline. An Azure Machine Learning pipeline can be as 
        //     * simple as one that calls a Python script, so may do just about anything.Pipelines 
        //     * should focus on machine learning tasks such as:
        //     * 
        //     * Data preparation including importing, validating and cleaning, munging and transformation, normalization, and staging
        //     * Training configuration including parameterizing arguments, filepaths, and logging / reporting configurations
        //     * Training and validating efficiently and repeatedly. Efficiency might come from specifying specific data subsets, 
        //     *      different hardware compute resources, distributed processing, and progress monitoring
        //     * Deployment, including versioning, scaling, provisioning, and access control
        //    ***/
        //    #endregion
        //    pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",
        //        inputColumnName: nameof(SentimentIssue.Text))
        //        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",
        //        featureColumnName: "Features"));

        //}
        //public Model(string trainDL)
        //{
        //    //Step 1: Create Context for testing and training
        //    mlContext = new MLContext();
        //    this.trainDataLocation = trainDataLocation;
        //    //Step 3: Convert input and output files into IDataView
        //    trainDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(trainDataLocation, hasHeader: true);
        //    //Step 4: Create pipeline
        //    #region pipelineExplanation 
        //    /** FROM https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
        //     * An Azure Machine Learning pipeline is an independently executable workflow 
        //     * of a complete machine learning task. Subtasks are encapsulated as a series 
        //     * of steps within the pipeline. An Azure Machine Learning pipeline can be as 
        //     * simple as one that calls a Python script, so may do just about anything.Pipelines 
        //     * should focus on machine learning tasks such as:
        //     * 
        //     * Data preparation including importing, validating and cleaning, munging and transformation, normalization, and staging
        //     * Training configuration including parameterizing arguments, filepaths, and logging / reporting configurations
        //     * Training and validating efficiently and repeatedly. Efficiency might come from specifying specific data subsets, 
        //     *      different hardware compute resources, distributed processing, and progress monitoring
        //     * Deployment, including versioning, scaling, provisioning, and access control
        //    ***/
        //    #endregion
        //    pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",
        //        inputColumnName: nameof(SentimentIssue.Text))
        //        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",
        //        featureColumnName: "Features"));
        //}
        #endregion
        public Model()
        {
            mlContext = new MLContext();
            pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                inputColumnName: nameof(SentimentIssue.Text))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label",
                featureColumnName: "Features"));
        }

        //trainDL: TrainDownloadLocation
        //public void Train(string trainDL)
        //{
        //    Console.WriteLine("Training...");
        //    trainDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(trainDL, hasHeader: true);
        //    trainedModel = pipeline.Fit(trainDataView);
        //    Console.WriteLine("Done Training...");
            
        //}
        public void Train()
        {
            Console.WriteLine("Training...");
            trainedModel = pipeline.Fit(trainDataView);
            Save();
            Console.WriteLine("Done Training...");
        }
        

        //sdl: saveDataLocation
        public void Save(string sDL)
        {
            mlContext.Model.Save(pipeline.Fit(trainDataView), trainDataView.Schema, sDL);
        }
        public void Save()
        {
            mlContext.Model.Save(pipeline.Fit(trainDataView), trainDataView.Schema, saveDataLocation);
        }
        public void Load(string modelLocation)
        {
            trainedModel = mlContext.Model.Load(modelLocation, out var modelInputSchema);
        }
        public void Load()
        {
            trainedModel = mlContext.Model.Load(saveDataLocation, out var modelInputSchema);
        }


        public void Test()
        {
            var predictions = trainedModel.Transform(testDataView);
            metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.Write(" tested for {0} accuracy.", metrics.Accuracy);
            //Console.WriteLine("Accuracy: {0}\n" +
            //                  "TestDataFile: {1}\n" +
            //                  "TrainDataFile: {2}\n" +
            //                  "TrainedModelLocation: {3}", metrics.Accuracy, testDataLocation, trainDataLocation, saveDataLocation);
            //Console.WriteLine("Done Testing, Press Any Key To Exit...");
           // Console.ReadKey();
        }

        public void Test(string testDL, string modelLocation)
        {
            Console.WriteLine("Testing...");
            Load(modelLocation);
            testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(testDL, hasHeader: true);
            var predictions = trainedModel.Transform(testDataView);
            metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("Accuracy: {0}\n" +
                              "TestDataFile: {1}\n" +
                              "TrainDataFile: {2}\n" +
                              "TrainedModelLocation: {3}", metrics.Accuracy, testDataLocation, trainDataLocation, modelLocation);
            Console.WriteLine("Done Testing, Press Any Key To Exit...");
            Console.ReadKey();
        }
        public Point TestGetPoint()
        {
            var predictions = trainedModel.Transform(testDataView);
            metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            //Console.Write(" tested for {0} accuracy.", metrics.Accuracy);
            int accuracy = Convert.ToInt32(metrics.Accuracy*100);
            return new Point(testFraction, accuracy);   
        }

        //public void TrainAndTest(string trainDL, string testDL, string sDL)
        //{
        //    Console.WriteLine("Training...");
        //    this.Train(trainDL);
        //    Console.WriteLine("Done Training...");
        //    Console.WriteLine("Testing...");
        //    this.Test(testDL, sDL);
        //    Console.WriteLine("Done Testing, Press Any Key To Exit...");
        //    Console.ReadKey();
        //}
        //public void TrainAndTest(string testDL)
        //{
        //    Console.WriteLine("Training...");
        //    this.Train();
        //    Console.WriteLine("Done Training...");
        //    Console.WriteLine("Testing...");
        //    this.Test(testDL, saveDataLocation);
        //    Console.WriteLine("Done Testing, Press Any Key To Exit...");
        //    Console.ReadKey();
        //}
        //public void TrainAndTest()
        //{
        //    Console.WriteLine("Training...");
        //    this.Train();
        //    Console.WriteLine("Done Training...");
        //    Console.WriteLine("Testing...");
        //    this.Test(testDataLocation, saveDataLocation);
        //    Console.WriteLine("Done Testing, Press Any Key To Exit...");
        //    Console.ReadKey();
        //}

        public void Predict(string feature)
        {
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);
            var prediction = predEngine.Predict(new SentimentIssue(feature)).Prediction;
            Console.WriteLine(prediction.ToString());
            Console.ReadLine();
        }

    }
}
