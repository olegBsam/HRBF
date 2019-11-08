using HRBF.neuralnet.behavior.layers;
using HRBF.neuralnet.model;
using System.Collections.Generic;
using System.Linq;

namespace HRBF.neuralnet.behavior
{
    public static class NeuronetBehavior
    {
        public static double Calculate(this HRBFNeuronet neuronet, double[] x)
        {
            var hiddenLayerOutput = neuronet.HiddenLayer.Calculate(x);
            var h = hiddenLayerOutput.Zip(neuronet.HiddenLayer.Neurons, (t1, t2) => t1 * t2.W).ToList();
            var output = neuronet.HiddenLayer.CalculateWeightedSum(hiddenLayerOutput);
            return output;
        }

        public static List<double> Learning(this HRBFNeuronet neuronet, List<Pair<double[], double>> learningSet, int epochCount, double learningCoef)
        {
            var errors = new List<double>();
            for (int epochInd = 0; epochInd < epochCount; epochInd++)
            {
                var xSet = learningSet[0].Item1.ToList();
                for (int learningSetInd = 0; learningSetInd < learningSet.Count; learningSetInd++)
                {
                    var y = neuronet.Calculate(xSet.ToArray());
                    var error = OneLearningIteration(neuronet, xSet.ToArray(), learningSet[learningSetInd].Item2, y, learningCoef);
                    errors.Add(error);
                    xSet.RemoveAt(0);
                    xSet.Add(y);
                }
            }
            return errors;
        }

        private static double OneLearningIteration(HRBFNeuronet neuronet, double[] xSet, double d, double y, double learningCoef)
        {
            var error = y - d;

            var newW = neuronet.HiddenLayer.RecalculatedW(error, xSet, learningCoef, neuronet.HiddenLayer.ActivationW);
            var newC = neuronet.HiddenLayer.RecalculatedC(error, xSet, learningCoef);
            var newQ = neuronet.HiddenLayer.RecalculatedQ(error, xSet, learningCoef);

            neuronet.HiddenLayer.SetParams(newW, newC, newQ);

            return error;
        }
    }
}
