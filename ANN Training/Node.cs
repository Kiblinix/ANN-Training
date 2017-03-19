using System;
using System.Collections.Generic;

class Node
{
    public List<Weight> inputs;

    public Node()
    {
        inputs = new List<Weight>();
    }

    public double delta;
    
    public double oldBias;
    private double? bias;
    public double? Bias
    {
        get
        {
            // If bias was never set, is likely to be an input node
            // So we set bias to 1 so it has no impact.
            return bias == null ? 1 : bias;
        }
        set
        {
            bias = value;
        }
    }

    private double output;
    public double Output
    {
        get
        {            
            return output;           
        }
        set
        {
            output = value;
        }
    }    

    public void CalculateOutput()
    {
        // Dot product of inputs and weights, plus bias.
        double sum = (double)Bias;
        for (int i = 0; i < inputs.Count; i++)
        {
            Node origin = inputs[i].origin;
            sum += origin.Output * inputs[i].value;
        }

        // Return sigmoid function result from dot product of inputs and weights, plus bias.
        Output = 1 / (1 + Math.Pow(Math.E, -sum));
    }

    public void BackwardsPass(double correctVal)
    {
        // Update delta of output node
        delta = (correctVal - Output) * (Output * (1 - Output));

        // Update delta of hidden nodes
        for (int i = 0; i < inputs.Count; i++)
        {
            Weight weight = inputs[i];
            Node origin = weight.origin;

            origin.delta = weight.value * delta * origin.Output * (1 - origin.Output);
        }        
    }

    public void UpdateWeights(double learningRate)
    {
        oldBias = (double)Bias;
        Bias = oldBias + learningRate * delta;

        // Add Momentum to Bias
        Bias = Bias + (0.9 * (Bias - oldBias));
        
        for (int i = 0; i < inputs.Count; i++)
        {
            Weight weight = inputs[i];            
            Node origin = weight.origin;

            weight.oldWeight = weight.value;
            weight.value = weight.oldWeight + learningRate * delta * origin.Output;

            // Add Momentum to Weight
            weight.value = weight.value + (0.9 * (weight.value - weight.oldWeight));
        }
    }

    public void UndoWeightChange()
    {
        Bias = oldBias;
        
        for (int i = 0; i < inputs.Count; i++)
        {
            Weight weight = inputs[i];
            weight.value = weight.oldWeight;
        }
    }
}
