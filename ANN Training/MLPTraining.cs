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
                Network network = new Network(i, cycles, 0.5);
                network.ExecuteNetwork();

                totalRMSE += network.GetRMSE();
                Console.WriteLine("Cycles : " + cycles + ", Actual Cycles: " + network.GetActualCycles() + ", Nodes: " + i + ", RMSE: " + network.GetRMSE());
            }
        }

        //for (int i = 0; i < 20; i++)
        //{
        //    Network network = new Network(5, 5000, 0.1);
        //    network.ExecuteNetwork();

        //    Console.WriteLine("Cycles : " + 5000 + ", Actual Cycles: " + network.GetActualCycles() + ", Nodes: " + i + ", RMSE: " + network.GetRMSE());
        //}

        //Network network = new Network(4, 20000, 0.1);
        //network.ExecuteNetwork();

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
        int numHiddenNodes = 2;
        double stepSize = 0.1;

        Network network = new Network(numHiddenNodes, 20000, stepSize);

        network.data = new List<List<double>>()
        {
            new List<double>() { 1, 0, 1 }
        };

        network.trainingSet = network.data;

        network.InitialiseNetwork();

        network.hiddenLayer[0].Bias = 1;
        network.hiddenLayer[0].inputs[0].value = 3;
        network.hiddenLayer[0].inputs[1].value = 4;

        network.hiddenLayer[1].Bias = -6;
        network.hiddenLayer[1].inputs[0].value = 6;
        network.hiddenLayer[1].inputs[1].value = 5;

        network.outputNode.Bias = -3.92;
        network.outputNode.inputs[0].value = 2;
        network.outputNode.inputs[1].value = 4;

        network.TrainNetwork();
    }
}