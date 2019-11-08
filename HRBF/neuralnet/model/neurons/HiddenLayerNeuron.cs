using HRBF.neuralnet.tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HRBF.neuralnet.model.neurons
{
    public class HiddenLayerNeuron
    {
        public double[] C { get; set; }
        public double[,] Q { get; set; }
        public double W { get; set; }

        public HiddenLayerNeuron(int n, double[] center)
        {
            C = center;
            Q = MatrixHelper.GetDiagmatrix(1.0, n);
            W = tools.MathHelper.Rnd.NextDouble();
        }
    }
}
