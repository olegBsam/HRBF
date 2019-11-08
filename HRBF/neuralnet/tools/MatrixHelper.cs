using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HRBF.neuralnet.tools
{
    public static class MatrixHelper
    {
        public static double[,] GetDiagmatrix(double diagElem, int n)
        {
            var result = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    result[i, j] = i == j ? diagElem : 0;
                }
            }
            return result;
        }

        public static double[] Sub(this double[] v1, double[] v2)
        {
            return v1.Zip(v2, (t1, t2) => t1 - t2).ToArray();
        }
    }
}
