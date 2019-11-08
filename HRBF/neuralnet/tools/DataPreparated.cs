using HRBF.neuralnet.model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HRBF.neuralnet.tools
{
    public static class DataPreparated
    {
        public static List<Pair<double[], double>> GetLearningSets(List<double> set, int inputSize)
        {
            var result = new List<Pair<double[], double>>();
            var data = set.ToArray();
            for (int i = 0; i < data.Length - inputSize; i++)
            {
                var xSet = new double[inputSize];
                Array.Copy(data, i, xSet, 0, inputSize);
                result.Add(new Pair<double[], double>(xSet, data[i + inputSize]));
            }
            return result;
        }

        internal static List<Pair<double[], double>> ExtractCenters(List<double> data, int inputNeuronCount, int hiddenNeuronCount)
        {
            var result = new List<Pair<double[], double>>();
            
            var allCenters = CentersExtractor.GetCenters(data.ToArray(), inputNeuronCount);

            for (int i = 0; i < allCenters.Count; i++)
            {
                var dist = allCenters.Select(t1 => CalcDist(t1, allCenters[i])).Max();
                if (!Contains(result, allCenters[i]))
                {
                    if (result.Count < hiddenNeuronCount)
                    {
                        result.Add(new Pair<double[], double>(allCenters[i], dist));
                    }
                    else if (result.Count >= hiddenNeuronCount)
                    {
                        var min = result.Min(t1 => t1.Item2);
                        var minItem = result.Find(t1 => t1.Item2 == min);
                        result.Remove(minItem);
                        result.Add(new Pair<double[], double>(allCenters[i], dist));
                    }
                }
            }

            return result;
        }

        private static bool Contains(List<Pair<double[], double>> result, double[] v)
        {
            for (int i = 0; i < result.Count; i++)
            {
                if (Equal1(result[i].Item1, v)) return true;
            }
            return false;
        }

        private static bool Equal1(double[] item1, double[] v)
        {
            for (int i = 0; i < item1.Length; i++)
            {
                if (item1[i] != v[i]) return false; 
            }
            return true;
        }

        private static double CalcDist(double[] v1, double[] v2)
        {
            var result = 0D;
            for (int i = 0; i < v1.Length; i++)
            {
                result += Math.Pow(v1[i] - v2[i], 2);
            }
            return Math.Sqrt(result);
        }

        /// <returns>(normalizeData, maxValue)</returns>
        public static (List<double> Data, double Max) NormalizedData(List<double> data)
        {
            var max = data.Max();
            return (data.Select(t1 => t1 / max).ToList(), max);
        }
            

        public static List<double> ExtractData(List<string> textData) =>
            textData.Select(t1 => double.Parse(t1.Replace('.', ','))).ToList();
    }
}
