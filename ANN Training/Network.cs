using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Text;

class Network
{
    private Random rand = new Random();

    public List<Node> inputLayer = new List<Node>();
    public List<Node> hiddenLayer = new List<Node>();
    public Node outputNode = new Node();

    public List<List<double>> data = new List<List<double>>();
    public List<List<double>> trainingSet = new List<List<double>>();
    private List<List<double>> validationSet = new List<List<double>>();
    private List<List<double>> testSet = new List<List<double>>();

    private int numHiddenNodes;   
    private double learningRate;
    private int numInputs;
    private int numCycles;
    private int actualCycles;

    private double previousError = 999;
    private double RMSE;

    private bool useKFold = true;
    
    // Used for de-normalising the output.
    private double outputColumnMin;
    private double outputColumnMax;

    public Network(int numHiddenNodes, int numCycles, double stepSize)
    {
        this.numHiddenNodes = numHiddenNodes;
        this.numCycles = numCycles;
        actualCycles = numCycles;
        learningRate = stepSize;
    }      

    public void ExecuteNetwork()
    {
        if (data.Count == 0)
        {
            ReadData("Data Set Exported.txt");
            ShuffleData();
            NormaliseData();

            if (!useKFold) SplitData();
            else trainingSet = data.ConvertAll(row => new List<double>(row));   // Deep copy of original data
        }

        InitialiseNetwork();
        if (!useKFold)
        {
            TrainNetwork();
            TestNetwork();
        }
        else
        {
            KFoldTrainNetwork();
            TestNetwork();
        }     
    }

    public void InitialiseNetwork()
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

    public void TrainNetwork()
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
                
                // Update weights and biases
                outputNode.UpdateWeights(learningRate);
                for (var j = 0; j < hiddenLayer.Count; j++)
                {
                    hiddenLayer[j].UpdateWeights(learningRate);
                }
            }

            // Every 500 epochs, test against validation set
            // If performance goes down, stop training.
            if (n % 500 == 0)
            {
                if (ValidateNetwork())
                {
                    actualCycles = n;
                    break;
                }
            }

            // Adjust learning rate each cycle
            learningRate = Annealing(n);            
        }
    }    
    
    private bool ValidateNetwork()
    {
        // Returns true if error has increased
        // Training should stop

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
        List<List<double>> testResults = new List<List<double>>();

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

            List<double> result = new List<double>()
            {
                predictedOutput,
                correctOutput
            };

            testResults.Add(result);
        }

        // Calculated Root Mean Squared Error
        RMSE = Math.Sqrt(totalError / testSet.Count);

        // Print output for interesting networks
        if (RMSE < 47)
        {
            PrintData(testResults, "results/nodes" + numHiddenNodes + "/RMSE" + ((int)RMSE).ToString() + ".csv");
        }
    }

    public void KFoldTrainNetwork()
    {
        int folds = 15;        

        for (int i = 0; i < folds; i++)
        {
            // Split data set 
            validationSet = trainingSet.GetRange(i * (trainingSet.Count / folds), trainingSet.Count / folds);
            trainingSet.RemoveRange(i * (trainingSet.Count / folds), trainingSet.Count / folds);

            for (int n = 0; n < numCycles / folds; n++)
            {
                for (int j = 0; j < trainingSet.Count; j++)
                {
                    List<double> row = trainingSet[j];

                    // Set input nodes to input values
                    for (var k = 0; k < inputLayer.Count; k++)
                    {
                        inputLayer[k].Output = row[k];
                    }

                    // Forward pass to hidden layer
                    for (var k = 0; k < hiddenLayer.Count; k++)
                    {
                        hiddenLayer[k].CalculateOutput();
                    }

                    // Forward pass to output node
                    outputNode.CalculateOutput();

                    // Do backwards pass and set delta for each node
                    // Carries through to hidden nodes
                    outputNode.BackwardsPass(row[row.Count - 1]);

                    // Update weights and biases
                    outputNode.UpdateWeights(learningRate);
                    for (var k = 0; k < hiddenLayer.Count; k++)
                    {
                        hiddenLayer[k].UpdateWeights(learningRate);
                    }
                }
                
                // Every 100 epochs, test against validation set
                // If performance goes down, stop training for this fold.
                // Let it train at least once
                if (n % 200 == 0 && n != 0)
                {
                    if (ValidateNetwork())
                    {
                        actualCycles = n;
                        break;
                    }
                }

                // Adjust learning rate each cycle
                learningRate = Annealing(n);
            }

            // Deep copy of original data so it doesn't get changed
            trainingSet = data.ConvertAll(row => new List<double>(row));
            //learningRate = 0.1;
        }

        // Test on whole data set at the end
        testSet = data;
    }

    private double RandomiseWeight()
    {
        // Randomise weight/bias based on number of inputs between -2/n and +2/n
        double offset = (double)2 / numInputs;
        return (rand.NextDouble() * (2 * offset)) - offset;
    }

    // Implement Annealing from lecture slides
    private double Annealing(int epoch)
    {
        double p = 0.05;
        double q = 0.5;
        double r = 3000;

        double fraction = 1 / (1 + Math.Pow(Math.E, 10 - (20 * (double)epoch / r)));

        return p + (q - p) * (1 - fraction);
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

    public void PrintData(List<List<double>> data, string filePath)
    {
        var csv = new StringBuilder();
        csv.AppendLine("Predicted Output, Correct Output");
        for (var i = 0; i < data.Count; i++)
        {
            List<double> row = data[i];
            string newLine = "";
            for (int j = 0; j < row.Count; j++)
            {
                newLine += row[j] + ",";
            }

            // Put some stats about the network
            if (i == 0) newLine += ",RMSE," + RMSE + ",";
            if (i == 1) newLine += ",Cycles," + numCycles + ",";
            if (i == 2) newLine += ",K Fold?," + useKFold + ",";

            newLine.TrimEnd(',');
            csv.AppendLine(newLine);
        }
        try
        {
            (new FileInfo(filePath)).Directory.Create();
            File.WriteAllText(filePath, csv.ToString());
        }
        catch
        {
            // File might be in use, don't care, do nothing.
        }
    }
}
