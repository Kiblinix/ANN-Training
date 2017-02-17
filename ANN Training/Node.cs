using System;
using System.Collections.Generic;

class Node
{
    public List<Weight> inputs;

    public Node()
    {
        inputs = new List<Weight>();
    }

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
        double sum = 0;
        for (int i = 0; i < inputs.Count; i++)
        {
            Node origin = inputs[i].origin;
            sum += (double)Bias + origin.Output * inputs[i].value;
        }

        // Return sigmoid function result from dot product of inputs and weights, plus bias.
        Output = 1 / (1 + Math.Pow(Math.E, -sum));
    }
}
