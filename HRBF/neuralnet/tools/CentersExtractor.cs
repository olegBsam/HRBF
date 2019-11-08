using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HRBF.neuralnet.tools
{
    public static class CentersExtractor
    {
        public static List<double[]> GetCenters(double[] sourceData, int cSize)
        {
            var centers = new List<double[]>();

            var halfCSize = cSize / 2;
            for (var i = halfCSize; i < sourceData.Length - halfCSize; i++)
            {
                if (i == halfCSize ||
                    i == sourceData.Length - halfCSize - 1 ||
                    sourceData[i] > sourceData[i + 1] && sourceData[i] > sourceData[i - 1] ||
                    sourceData[i] < sourceData[i + 1] && sourceData[i] < sourceData[i - 1])
                {
                    centers.Add(GetCVector(sourceData, i, cSize));
                }
            }
            return centers;
        }

        private static double[] GetCVector(double[] sourceData, int cCenterIndex, int cSize)
        {
            var center = new double[cSize];
            var halfSize = cSize / 2;
            Array.Copy(sourceData, cCenterIndex - halfSize, center, 0, cSize);
            return center;
        }
    }
}
