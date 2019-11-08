using Accord.Math;
using HRBF.neuralnet.behavior.neurons;
using HRBF.neuralnet.model.layers;
using HRBF.neuralnet.model.neurons;
using System;
using System.Collections.Generic;
using System.Linq;

namespace HRBF.neuralnet.behavior.layers
{
    public static class HiddenLayerExt
    {
        public static double[] Calculate(this HiddenLayer layer,  double[] x) =>
            layer.Neurons
                .Select(neuron => (neuron).CalculateOut(x))
                .ToArray();
        public static double CalculateWeightedSum(this HiddenLayer layer, double[] output) =>
           layer.Neurons
               .Zip(output, (t1, t2) => t1.W * t2).Sum() + layer.ActivationW;

        public static List<double> RecalculatedW(this HiddenLayer layer, double error, double[] xSet, double learningCoef, double activationW) =>
            (new List<double> { activationW - learningCoef * error}).Concat(
                layer.Neurons
                    .Select(neuron => neuron.W - learningCoef * neuron.dEdWi(xSet, error))
                    .ToList())
            .ToList();
           
        public static List<double[]> RecalculatedC(this HiddenLayer layer, double error, double[] xSet, double learningCoef) =>
            layer.Neurons
                .Select((neuron, i) =>
                    neuron.C
                        .Select((c, j) => c - learningCoef * neuron.dEdCij(error, xSet, j))
                        .ToArray())
                .ToList();

        public static List<List<List<double>>> RecalculatedQ(this HiddenLayer layer, double error, double[] xSet, double learningCoef)
        {
            var k = layer.Neurons
                .Select((neuron, i) =>
                    neuron.Q.GetColumn(0)
                        .Select((_, j) =>
                            neuron.Q.GetRow(j)
                                .Select((q, r) =>
                                    q - learningCoef * neuron.dEdQijr(xSet, error, j, r))
                                .ToList())
                        .ToList())
                .ToList();
            return k;
        }
            //layer.Neurons
            //    .Select((neuron, i) =>
            //        neuron.Q.GetColumn(0)
            //            .Select((_, j) =>
            //                neuron.Q.GetRow(j)
            //                    .Select((q, r) =>
            //                        q - learningCoef * neuron.dEdQijr(xSet, error, j, r))
            //                    .ToList())
            //            .ToList())
            //    .ToList();

        public static void SetParams(this HiddenLayer layer, List<double> newW, List<double[]> newC, List<List<List<double>>> newQ)
        {
            layer.ActivationW = newW[0];
            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                layer.Neurons[i].W = newW[i + 1];
                layer.Neurons[i].C = newC[i];

                for (int j = 0; j < newQ[i].Count; j++)
                {
                    for (int r = 0; r < newQ[i].Count; r++)
                    {
                        layer.Neurons[i].Q[j, r] = newQ[i][j][r];
                    }
                }
            }
        }
        
    }
}
