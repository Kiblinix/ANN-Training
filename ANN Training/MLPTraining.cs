using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

class MLPTraining
{
    static void Main(string[] args)
    {
        //testAll();
        //return;        

        // Import Data
        List<List<double>> trainingData = readData();
        printData(trainingData, "imported.csv");
        trainingData = normaliseData(trainingData);
        printData(trainingData, "normalised.csv");

        Network firstNetwork = new Network(trainingData, 8, 2000, 0.1);
        firstNetwork.ExecuteNetwork();

        Console.WriteLine("Press any key to exit.");
        Console.Read();
    }

    static List<List<double>> readData()
    {
        List<List<double>> trainingData = new List<List<double>>();

        using (StreamReader reader = new StreamReader("../../CWDataStudent.txt"))
        {
            string row;
            while ((row = reader.ReadLine()) != null)
            {
                // Split tab delimited rows.
                string[] splitRow = row.Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
                List<double> data = new List<double>();

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
                        data.Add(field);           
                }

                if (!failed) { trainingData.Add(data); }
            }

            reader.Close();
        }

        return trainingData;
    }

    static List<List<double>> normaliseData(List<List<double>> data)
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

        return normalisedData;
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