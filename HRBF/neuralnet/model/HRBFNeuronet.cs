using HRBF.neuralnet.model.layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HRBF.neuralnet.model
{
    public class HRBFNeuronet
    {
        public int K { get; private set; }
        public int N { get; private set; }

        public HiddenLayer HiddenLayer { get; private set; }

        public HRBFNeuronet(int k, int n, List<double[]> centers)
        {
            K = k; N = n;
            HiddenLayer = new HiddenLayer(k, n, centers);
        }
    }
}
