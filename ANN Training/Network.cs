using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class Network
{
    private Random rand = new Random();

    private List<Node> inputLayer = new List<Node>();
    private List<Node> hiddenLayer = new List<Node>();
    private Node outputNode = new Node();

    private List<List<double>> data = new List<List<double>>();
    private int numHiddenNodes;
    private int numCycles;
    private double stepSize;
    private int numInputs;    

    public Network(int numHiddenNodes, int numCycles, double stepSize)
    {
        this.numHiddenNodes = numHiddenNodes;
        this.numCycles = numCycles;
        this.stepSize = stepSize;
    }

    public void ExecuteNetwork()
    {
        ReadData("../../CWDataStudent.txt");
        NormaliseData();

        InitialiseNetwork();
        TrainNetwork();
        TestNetwork();
    }    

    private void InitialiseNetwork()
    {
        for (int i = 0; i < numInputs; i++)
        {
            Node input = new Node();
            inputLayer.Add(input);
        }

        for (int i = 0; i < numHiddenNodes; i++)
        {
            Node hiddenNode = new Node();
            hiddenNode.Bias = RandomiseWeight(numInputs);

            for (var j = 0; j < inputLayer.Count; j++)
            {
                Weight inputWeight = new Weight(inputLayer[j], RandomiseWeight(numInputs));
                hiddenNode.inputs.Add(inputWeight);
            }

            hiddenLayer.Add(hiddenNode);
        }
        
        outputNode.Bias = RandomiseWeight(numInputs);
        for (var i = 0; i < hiddenLayer.Count; i++)
        {
            Weight inputWeight = new Weight(hiddenLayer[i], RandomiseWeight(numInputs));
            outputNode.inputs.Add(inputWeight);
        }
    }

    private void TrainNetwork()
    {
        for (int n = 0; n < numCycles; n++)
        {
            double sum = 0;
            for (int i = 0; i < data.Count; i++)
            {
                List<double> row = data[i];

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
                outputNode.UpdateWeights(stepSize);
                for (var j = 0; j < hiddenLayer.Count; j++)
                {
                    hiddenLayer[j].UpdateWeights(stepSize);
                }
                
                sum += Math.Pow(outputNode.Output - row[row.Count - 1], 2);
            }

            double avgError = sum / data.Count;
        }
    }

    private void TestNetwork()
    {
        double totalError = 0;
        for (int i = 0; i < data.Count; i++)
        {
            List<double> row = data[i];

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

            totalError += Math.Pow(outputNode.Output - row[row.Count - 1], 2);
        }

        double meanSquaredError = totalError / data.Count;

        Console.WriteLine(meanSquaredError);
    }

    private double RandomiseWeight(int numInputs)
    {
        // Randomise weight/bias based on number of inputs between -2/n and +2/n
        return (rand.NextDouble() * 2 / numInputs) - 2 / numInputs;
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

                    if (i != 4)   // Ignore columns 2 and 3 as they have low correlation
                        inputs.Add(field);
                }

                if (!failed) { data.Add(inputs); }
            }

            reader.Close();
        }

        numInputs = data[0].Count - 1;
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

                // Normalise between 0.1 and 0.9
                newRow.Add(((row[j] - columnMin) / (columnMax - columnMin)) * 0.8 + 0.1);
            }

            normalisedData.Add(newRow);
        }

        data = normalisedData;
    }
}
