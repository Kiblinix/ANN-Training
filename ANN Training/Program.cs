﻿using System;

class Program
{
    static void Main(string[] args)
    {
        double[][] trainingData = {
            new double[] { 1, 1, 4, -1 },
            new double[] { 1, 2, 9, 1 },
            new double[] { 1, 5, 6, 1 },
            new double[] { 1, 4, 5, 1 },
            new double[] { 1, 6, 0.7, -1 },
            new double[] { 1, 1, 1.5, -1 },
        };
        double[] weights = { 0, 0, 0 };

        // PerceptronLearning(trainingData, weights);
        AdalineLearning(trainingData, weights);

        Console.WriteLine("");
        Console.WriteLine("Bias: " + weights[0]);
        Console.WriteLine("w1:   " + weights[1]);
        Console.WriteLine("w2:   " + weights[2]);
        Console.WriteLine("");
        Console.WriteLine("Press any key to exit.");
        Console.Read();
    }

    static double ThresholdFunction(double sum, double threshold)
    {
        double output;

        if (sum > threshold)
            output = 1;
        else if (sum < threshold)
            output = -1;
        else
            output = 0;

        return output;
    }

    static void AdalineLearning(double[][] trainingData, double[] weights)
    {
        // Randomise Weights
        Random rand = new Random();
        weights[0] = rand.NextDouble();
        weights[1] = rand.NextDouble();
        weights[2] = rand.NextDouble();

        double p = 0.01;

        int cycles = 500;
        for (int i = 0; i < cycles; i++)
        {
            for (int j = 0; j < trainingData.Length; j++)
            {
                double[] row = trainingData[j];
                double S = (weights[0] * row[0]) + (weights[1] * row[1]) + (weights[2] * row[2]);
                weights[0] = weights[0] + (p * (row[3] - S) * row[0]);
                weights[1] = weights[1] + (p * (row[3] - S) * row[1]);
                weights[2] = weights[2] + (p * (row[3] - S) * row[2]);
            }
        }
    }

    static void PerceptronLearning(double[][] trainingData, double[] weights)
    {
        double threshold = 0;
        bool failedTest = false;

        do
        {
            failedTest = false;

            for (int i = 0; i < trainingData.Length; i++)
            {
                double[] row = trainingData[i];
                double S = (weights[0] * row[0]) + (weights[1] * row[1]) + (weights[2] * row[2]);

                if (ThresholdFunction(S, threshold) != row[3])
                {
                    failedTest = true;
                    weights[0] = weights[0] + (row[3] * row[0]);
                    weights[1] = weights[1] + (row[3] * row[1]);
                    weights[2] = weights[2] + (row[3] * row[2]);
                }
            }
        } while (failedTest);
    }
}