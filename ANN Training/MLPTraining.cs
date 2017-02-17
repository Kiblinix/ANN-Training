using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

class MLPTraining
{
    static Random rand = new Random();

    static void Main(string[] args)
    {
        // Set network attributes
        List<List<double>> trainingData = readData();
        trainingData = normaliseData(trainingData);        
        int numHiddenNodes = 10;
        double stepSize = 0.01;
        int numInputs = trainingData[0].Count - 1;

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
            hiddenNode.Bias = randomiseWeight(numInputs);

            for (var j = 0; j < inputLayer.Count; j++)
            {
                Weight inputWeight = new Weight(inputLayer[j], randomiseWeight(numInputs));
                hiddenNode.inputs.Add(inputWeight);
            }

            hiddenLayer.Add(hiddenNode);
        }

        Node outputNode = new Node();
        outputNode.Bias = randomiseWeight(numInputs);
        for (var i = 0; i < hiddenLayer.Count; i++)
        {
            Weight inputWeight = new Weight(hiddenLayer[i], randomiseWeight(numInputs));
            outputNode.inputs.Add(inputWeight);            
        }

        // Loop through data
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
        }

        Console.WriteLine(outputNode.Output);
        
        //Console.WriteLine("");
        //Console.WriteLine("Bias: " + weights[0]);
        //Console.WriteLine("w1:   " + weights[1]);
        //Console.WriteLine("w2:   " + weights[2]);
        //Console.WriteLine("");
        Console.WriteLine("Press any key to exit.");
        Console.Read();
    }

    static double randomiseWeight(int numInputs)
    {
        // Randomise weight/bias based on number of inputs between -2/n and +2/n        
        double val = (rand.NextDouble() * 4 * numInputs) - 2 * numInputs;        
        return val;
    }

    static List<List<double>> readData()
    {
        List<List<double>> trainingData = new List<List<double>>();

        using (StreamReader reader = new StreamReader("../../CWDataStudent2.txt"))
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
        for (var i = 0; i < data.Count; i++)
        {
            List<double> row = data[i];
            for (int j = 0; j < row.Count - 1; j++)     // Ignore correct output column.
            {
                double columnMax = data.Max(u => u[j]);
                double columnMin = data.Min(u => u[j]);

                // Normalise between 0.1 and 0.9
                row[j] = ((row[j] - columnMin) / (columnMax - columnMin)) * 0.8 + 0.1;
            }
        }

        return data;
    }
}