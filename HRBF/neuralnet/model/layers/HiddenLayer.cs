using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HRBF.neuralnet.model.neurons;
using HRBF.neuralnet.tools;

namespace HRBF.neuralnet.model.layers
{
    public class HiddenLayer
    {
        public int K { get; private set; }
        public int N { get; private set; }

        public HiddenLayerNeuron[] Neurons { get; set; }
        public double ActivationW { get; internal set; }

        public HiddenLayer(int k, int n, List<double[]> centers)
        {
            ActivationW = 0.2;// tools.MathHelper.Rnd.NextDouble();
            K = k; N = n;
            Neurons = new HiddenLayerNeuron[k];
            for (int i = 0; i < k; i++)
            {
                Neurons[i] = new HiddenLayerNeuron(n, centers[i]);
            }
        }
    }
}
