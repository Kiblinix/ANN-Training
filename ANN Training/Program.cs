using System;

class Program
{
    static void Main(string[] args)
    {
        double[][] trainingData = {
            new double[] { 1, 1, 8, 1 },
            new double[] { 1, 6, 2, -1 }
        };
        double[] weights = { 0, 0, 0 };
        double threshold = 0;

        bool failedTest = false;
        do
        {
            failedTest = false;

            for (int i = 0; i < trainingData.Length; i++)
            {
                double[] row = trainingData[i];
                double S = (weights[0] * row[0]) + (weights[1] * row[1]) + (weights[2] * row[2]);
                double output;

                if (S > threshold)
                    output = 1;
                else if (S < threshold)
                    output = -1;
                else
                    output = 0;

                if (output != row[3])
                {
                    failedTest = true;
                    weights[0] = weights[0] + (row[3] * row[0]);
                    weights[1] = weights[1] + (row[3] * row[1]);
                    weights[2] = weights[2] + (row[3] * row[2]);
                }
            }
        } while (failedTest);

        Console.WriteLine("");
        Console.WriteLine("Bias: " + weights[0]);
        Console.WriteLine("w1:   " + weights[1]);
        Console.WriteLine("w2:   " + weights[2]);
        Console.WriteLine("");
        Console.WriteLine("Press any key to exit.");
        Console.Read();
    }
}