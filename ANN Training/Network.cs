using System;
using System.Collections.Generic;

class Network
{
    private Random rand = new Random();

    private List<Node> inputLayer = new List<Node>();
    private List<Node> hiddenLayer = new List<Node>();
    private Node outputNode = new Node();

    private List<List<double>> trainingData;
    private int numHiddenNodes;
    private int numCycles;
    private double stepSize;
    private int numInputs;    

    public Network(List<List<double>> trainingData, int numHiddenNodes, int numCycles, double stepSize)
    {
        this.trainingData = trainingData;
        this.numHiddenNodes = numHiddenNodes;
        this.numCycles = numCycles;
        this.stepSize = stepSize;

        numInputs = trainingData[0].Count - 1;
    }

    public void ExecuteNetwork()
    {
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
            for (int i = 0; i < trainingData.Count; i++)
            {
                List<double> row = trainingData[i];

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

            double avgError = sum / trainingData.Count;
        }
    }

    private void TestNetwork()
    {
        double totalError = 0;
        for (int i = 0; i < trainingData.Count; i++)
        {
            List<double> row = trainingData[i];

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

        double meanSquaredError = totalError / trainingData.Count;

        Console.WriteLine(meanSquaredError);
    }

    private double RandomiseWeight(int numInputs)
    {
        // Randomise weight/bias based on number of inputs between -2/n and +2/n
        return (rand.NextDouble() * 2 / numInputs) - 2 / numInputs;
    }
}
