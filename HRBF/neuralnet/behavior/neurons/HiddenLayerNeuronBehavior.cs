using Accord.Math;
using System;
using System.Linq;
using HRBF.neuralnet.tools;
using HRBF.neuralnet.model.neurons;

namespace HRBF.neuralnet.behavior.neurons
{
    public static class HiddenLayerNeuronExt
    {
        private const double COEF = -1.0 / 2.0;
        public static double CalculateOut(this HiddenLayerNeuron neuron, double[] x)
        {
            if (neuron is HiddenLayerNeuron)
            {
                var v1 = neuron.Q.Dot(x.Sub(neuron.C));
                return Math.Exp(COEF * v1.Dot(v1.Transpose())[0]);
            }
            else return 1;
        }

        private static double CalculationZr(HiddenLayerNeuron neuron, double[] x, int r)
        {
            var result = 0D;
            for (int j = 0; j < neuron.Q.GetLength(0); j++)
            {
                result += neuron.Q[j, r] * (x[j] - neuron.C[j]);
            }
            return result;
        }

        private static double CalculationUi(HiddenLayerNeuron neuron, double[] x)
        {
            var result = 0D;
            for (int j = 0; j < neuron.Q.GetLength(0); j++)
            {
                result += Math.Pow(CalculationZr(neuron, x, j), 2);
            }

            return result;
           
        }

        public static double dEdWi(this HiddenLayerNeuron neuron, double[] x, double error) =>
            Math.Exp(COEF * CalculationUi(neuron, x)) * error;

        public static double dEdCij(this HiddenLayerNeuron neuron, double error, double[] x, int j) =>
             -dEdWi(neuron, x, error) * neuron.W * neuron.Q.GetRow(j)
                    .Select((t1, r) => t1 * CalculationZr(neuron, x, r))
                    .Sum();

        public static double dEdQijr(this HiddenLayerNeuron neuron, double[] x, double error, int j, int r) =>
             -dEdWi(neuron, x, error) * neuron.W * (x[j] - neuron.C[j]) * CalculationZr(neuron, x, r);
        

    }
}
