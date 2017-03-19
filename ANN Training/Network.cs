using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;

class Network
{
    private Random rand = new Random();

    private List<Node> inputLayer = new List<Node>();
    private List<Node> hiddenLayer = new List<Node>();
    private Node outputNode = new Node();

    private List<List<double>> data = new List<List<double>>();
    private List<List<double>> trainingSet = new List<List<double>>();
    private List<List<double>> validationSet = new List<List<double>>();
    private List<List<double>> testSet = new List<List<double>>();

    private int numHiddenNodes;   
    private double stepSize;
    private int numInputs;
    private int numCycles;
    private int actualCycles;

    private double previousError = 999;
    private double RMSE;

    private bool useBoldDriver = false;
    
    // Used for de-normalising the output.
    private double outputColumnMin;
    private double outputColumnMax;

    public Network(int numHiddenNodes, int numCycles, double stepSize)
    {
        this.numHiddenNodes = numHiddenNodes;
        this.numCycles = numCycles;
        actualCycles = numCycles;
        this.stepSize = stepSize;
    }      

    public void ExecuteNetwork()
    {
        if (data.Count == 0)
        {
            ReadData("../../CWDataStudent.txt");
            ShuffleData();
            NormaliseData();
            SplitData();
        }        

        InitialiseNetwork();
        TrainNetwork();
        TestNetwork();        
    }

    private void ShuffleData()
    {
        // Randomly shuffle input data
        // Before splitting into Training, Validation and Test sets
        // Based on Fisher-Yates Shuffle

        int n = data.Count;
        while (n > 1)
        {
            n--;
            int k = rand.Next(n + 1);
            List<Double> value = data[k];
            data[k] = data[n];
            data[n] = value;
        }
    }

    private void InitialiseNetwork()
    {
        numInputs = data[0].Count - 1;
        for (int i = 0; i < numInputs; i++)
        {
            Node input = new Node();
            inputLayer.Add(input);
        }

        for (int i = 0; i < numHiddenNodes; i++)
        {
            Node hiddenNode = new Node();
            hiddenNode.Bias = RandomiseWeight();

            for (var j = 0; j < inputLayer.Count; j++)
            {
                Weight inputWeight = new Weight(inputLayer[j], RandomiseWeight());
                hiddenNode.inputs.Add(inputWeight);
            }

            hiddenLayer.Add(hiddenNode);
        }
        
        outputNode.Bias = RandomiseWeight();
        for (var i = 0; i < hiddenLayer.Count; i++)
        {
            Weight inputWeight = new Weight(hiddenLayer[i], RandomiseWeight());
            outputNode.inputs.Add(inputWeight);
        }
    }

    private void TrainNetwork()
    {
        for (int n = 0; n < numCycles; n++)
        {
            for (int i = 0; i < trainingSet.Count; i++)
            {
                List<double> row = trainingSet[i];

                // Set input nodes to input values
                for (var j = 0; j < inputLayer.Count; j++)
                {
                    inputLayer[j].Output = row[j];
                }

                // Forward pass to hidden layer
                for (var j = 0; j < hiddenLayer.Count; j++)
                {
                    hiddenLayer[j].CalculateOutput();
                }

                // Forward pass to output node
                outputNode.CalculateOutput();
                
                // Do backwards pass and set delta for each node
                // Carries through to hidden nodes
                outputNode.BackwardsPass(row[row.Count - 1]);

                // Compare output of network to actual value, get error
                // For bold driver
                //double predictedOutput = ((outputNode.Output - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
                //double correctOutput = ((row[row.Count - 1] - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
                //double beforeError = Math.Sqrt(Math.Pow((predictedOutput - correctOutput), 2));
                double beforeError = row[row.Count - 1] - outputNode.Output;

                // Update weights and biases
                outputNode.UpdateWeights(stepSize);
                for (var j = 0; j < hiddenLayer.Count; j++)
                {
                    hiddenLayer[j].UpdateWeights(stepSize);
                }

                // Compare new output of network to actual value, get error
                // For bold driver
                
                if (useBoldDriver)
                {
                    // Forward pass to hidden layer
                    for (var j = 0; j < hiddenLayer.Count; j++)
                    {
                        hiddenLayer[j].CalculateOutput();
                    }

                    // Forward pass to output node
                    outputNode.CalculateOutput();

                    //double predictedOutput2 = ((outputNode.Output - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
                    //correctOutput = ((row[row.Count - 1] - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
                    //double afterError = Math.Sqrt(Math.Pow((predictedOutput2 - correctOutput), 2));
                    double afterError = row[row.Count - 1] - outputNode.Output;

                    // Bold Driver
                    if (afterError > beforeError)
                    {
                        // Learning rate was too large
                        stepSize *= 0.5;

                        // Undo weight changes
                        outputNode.UndoWeightChange();
                        for (var j = 0; j < hiddenLayer.Count; j++)
                        {
                            hiddenLayer[j].UndoWeightChange();
                        }

                        // Try same row again
                        i -= 1;
                    }
                    else
                    {
                        // Learning rate may be too low
                        stepSize *= 1.1;
                        if (stepSize > 2) stepSize = 2;
                    }
                }                
            }

            // Every 100 epochs, test against validation set
            // If performance goes down, stop training.
            if (n % 500 == 0)
            {
                if (ValidateNetwork())
                {
                    actualCycles = n;
                    break;
                }
            }

            //if (n % 500 == 499) Console.WriteLine(n + 1 + " passes complete.");
        }
    }

    private bool ValidateNetwork()
    {
        // Returns true if error has increased
        // Testing should stop

        double totalError = 0;

        for (int i = 0; i < validationSet.Count; i++)
        {
            List<double> row = validationSet[i];

            // Set input nodes to input values
            for (var j = 0; j < inputLayer.Count; j++)
            {
                inputLayer[j].Output = row[j];
            }

            // Forward pass to hidden layer
            for (var j = 0; j < hiddenLayer.Count; j++)
            {
                hiddenLayer[j].CalculateOutput();
            }

            // Forward pass to output node
            outputNode.CalculateOutput();

            // Undo the data normalisation back to the previous min/max range
            double predictedOutput = ((outputNode.Output - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
            double correctOutput = ((row[row.Count - 1] - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
            totalError += Math.Pow((predictedOutput - correctOutput), 2);
        }

        double currentError = Math.Sqrt(totalError / validationSet.Count);

        //Console.WriteLine("Err: " + currentError);

        if (currentError > previousError)
        {
            return true;
        }

        previousError = currentError;
        return false;
    }

    private void TestNetwork()
    {       
        double totalError = 0;

        for (int i = 0; i < testSet.Count; i++)
        {
            List<double> row = testSet[i];

            // Set input nodes to input values
            for (var j = 0; j < inputLayer.Count; j++)
            {
                inputLayer[j].Output = row[j];
            }

            // Forward pass to hidden layer
            for (var j = 0; j < hiddenLayer.Count; j++)
            {
                hiddenLayer[j].CalculateOutput();
            }

            // Forward pass to output node
            outputNode.CalculateOutput();

            // Undo the data normalisation back to the previous min/max range
            double predictedOutput = ((outputNode.Output - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
            double correctOutput = ((row[row.Count - 1] - 0.1) / 0.8) * (outputColumnMax - outputColumnMin) + outputColumnMin;
            
            totalError += Math.Pow((predictedOutput - correctOutput), 2);
        }

        // Calculated Root Mean Squared Error
        RMSE = Math.Sqrt(totalError / testSet.Count);

        //Console.WriteLine("RMSE: " + RMSE);
    }

    private double RandomiseWeight()
    {
        // Randomise weight/bias based on number of inputs between -2/n and +2/n
        double offset = (double)2 / numInputs;
        return (rand.NextDouble() * (2 * offset)) - offset;
    }

    private void ReadData(string path)
    {
        using (StreamReader reader = new StreamReader(path))
        {
            string row;
            while ((row = reader.ReadLine()) != null)
            {
                // Split tab delimited rows.
                string[] splitRow = row.Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
                List<double> inputs = new List<double>();

                bool failed = false;
                for (var i = 0; i < splitRow.Length; i++)
                {
                    // If conversion to double fails, skip whole row.
                    double field;
                    if (!Double.TryParse(splitRow[i], out field) || field == -999)
                    {
                        failed = true;
                    }

                    //if (i != 4)   // Ignore columns 2 and 3 as they have low correlation
                        inputs.Add(field);
                }

                if (!failed) { data.Add(inputs); }
            }

            reader.Close();
        }
    }

    private void NormaliseData()
    {
        List<List<double>> normalisedData = new List<List<double>>();
        for (var i = 0; i < data.Count; i++)
        {
            List<double> row = data[i];
            List<double> newRow = new List<double>();

            for (int j = 0; j < row.Count; j++)
            {
                double columnMax = data.Max(u => u[j]);
                double columnMin = data.Min(u => u[j]);

                if (j == row.Count - 1)
                {
                    outputColumnMax = columnMax;
                    outputColumnMin = columnMin;
                }

                // Normalise between 0.1 and 0.9
                newRow.Add(((row[j] - columnMin) / (columnMax - columnMin)) * 0.8 + 0.1);
            }

            normalisedData.Add(newRow);
        }

        data = normalisedData;
    }

    private void SplitData()
    {
        int trainingAmount = (int)(0.6 * data.Count);
        int validationAmount = (data.Count - trainingAmount) / 2;
        int testAmount = data.Count - trainingAmount - validationAmount;

        trainingSet = data.GetRange(0, trainingAmount);
        validationSet = data.GetRange(trainingAmount, validationAmount);
        testSet = data.GetRange(trainingAmount + validationAmount, testAmount);
    }

    public int GetHiddenNodes()
    {
        return numHiddenNodes;
    }

    public int GetNumCycles()
    {
        return numCycles;
    }

    public double GetRMSE()
    {
        return RMSE;
    }

    public double GetActualCycles()
    {
        return actualCycles;
    }
}
