using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

class MLPTraining
{
    static void Main(string[] args)
    {
        // Train Networks with different numbers of hidden nodes
        Console.WriteLine("Cycles, Nodes, AVG RMSE, MIN RMSE, MAX RMSE");
        var csv = new StringBuilder();
        csv.AppendLine("Cycles, Nodes, AVG RMSE, MIN RMSE, MAX RMSE");

        for (int i = 2; i <= 12; i++)
        {
            // Do network 10 times each and take AVG RMSE
            double totalRMSE = 0;
            int cycles = 50000;

            double minRMSE = 99;
            double maxRMSE = 0;

            for (int j = 0; j < 10; j++)
            {
                Network network = new Network(i, cycles, 0.5);
                network.ExecuteNetwork();

                totalRMSE += network.GetRMSE();

                if (network.GetRMSE() > maxRMSE) maxRMSE = network.GetRMSE();
                if (network.GetRMSE() < minRMSE) minRMSE = network.GetRMSE();

                //Console.WriteLine("Cycles : " + cycles + ", Actual Cycles: " + network.GetActualCycles() + ", Nodes: " + i + ", RMSE: " + network.GetRMSE());
            }

            Console.WriteLine(cycles + " , " + i + " , " + totalRMSE / 10 + " , " + minRMSE + " , " + maxRMSE);
            csv.AppendLine(cycles + "," + i + "," + totalRMSE / 10 + "," + minRMSE + "," + maxRMSE);
        }

        try
        {
            File.WriteAllText("networkInfo.csv", csv.ToString());
        }
        catch
        {
            // File might be in use, don't care, do nothing.
        }

        Console.WriteLine("Press any key to exit.");
        Console.ReadKey();
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