using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

class MLPTraining
{
    static void Main(string[] args)
    {
        // Train Networks with different numbers of hidden nodes
        for (int i = 2; i <= 12; i++)
        {
            // Do network 10 times each and take AVG RMSE
            double totalRMSE = 0;
            int cycles = 10000;

            for (int j = 0; j < 10; j++)
            {
                Network network = new Network(i, cycles, 0.1);
                network.ExecuteNetwork();

                totalRMSE += network.GetRMSE();
                Console.WriteLine("Cycles : " + cycles + ", Actual Cycles: " + network.GetActualCycles() + ", Nodes: " + i + ", RMSE: " + network.GetRMSE());
            }            
        }
        
        Console.WriteLine("Press any key to exit.");
        Console.Read();
    }
    
    static void printData(List<List<double>> data, string filePath)
    {
        var csv = new StringBuilder();
        for (var i = 0; i < data.Count; i++)
        {
            List<double> row = data[i];
            string newLine = "";
            for (int j = 0; j < row.Count; j++)
            {
                newLine += row[j] + ",";
            }
            newLine.TrimEnd(',');
            csv.AppendLine(newLine);
        }
        try
        {
            File.WriteAllText(filePath, csv.ToString());
        }
        catch
        {
            // File might be in use, don't care, do nothing.
        }
    }

    static void testAll()
    {
        List<List<double>> data = new List<List<double>>()
        {
            new List<double>() { 1, 0, 1 }
        };

        int numHiddenNodes = 2;
        double stepSize = 0.1;
        int numInputs = data[0].Count - 1;

        // Initialise Network
        List<Node> inputLayer = new List<Node>();
        for (int i = 0; i < numInputs; i++)
        {
            Node input = new Node();
            inputLayer.Add(input);
        }

        List<Node> hiddenLayer = new List<Node>();
        for (int i = 0; i < numHiddenNodes; i++)
        {
            Node hiddenNode = new Node();

            double weight = 1;
            if (i == 1) weight = -6;
            hiddenNode.Bias = weight;

            for (var j = 0; j < inputLayer.Count; j++)
            {
                double weight2 = 3;
                if (i == 0 & j == 1) weight2 = 4;
                if (i == 1 & j == 0) weight2 = 6;
                if (i == 1 & j == 1) weight2 = 5;

                Weight inputWeight = new Weight(inputLayer[j], weight2);
                hiddenNode.inputs.Add(inputWeight);
            }

            hiddenLayer.Add(hiddenNode);
        }

        Node outputNode = new Node();
        outputNode.Bias = -3.92;
        for (var i = 0; i < hiddenLayer.Count; i++)
        {
            double weight2 = 2;
            if (i == 1) weight2 = 4;
            Weight inputWeight = new Weight(hiddenLayer[i], weight2);
            outputNode.inputs.Add(inputWeight);
        }

        for (int n = 0; n < 500; n++)
        {
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
             }
        }

        double output = outputNode.Output;
        
    }
}