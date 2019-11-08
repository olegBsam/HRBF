using HRBF.neuralnet.model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HRBF.neuralnet.tools;
using HRBF.neuralnet.behavior;
using System.IO;

namespace HRBF
{
    class Test
    {
        internal void Start()
        {
            test2();
        }

        public void test2()
        {
            var hiddenNeuronCount = 5;
            var inputNeuronCount = 5;

            var data =  MeteoData.Read();
            data = DataPreparated.NormalizedData(data).Data;
            var testdata = DataPreparated.GetLearningSets(data, inputNeuronCount);
            var centers = DataPreparated.ExtractCenters(data, inputNeuronCount, hiddenNeuronCount);


            var nn = new HRBFNeuronet(hiddenNeuronCount, inputNeuronCount, centers.Select(t1 => t1.Item1).ToList());
            nn.Learning(testdata, 200, 0.02);

            var d = testdata.Select(o => o.Item2).ToList();
            var y = testdata.Select(o => nn.Calculate(o.Item1)).ToList();

            var l = nn.Calculate(testdata[200].Item1);
        }
    }
}
