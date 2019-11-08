using HRBF.neuralnet.behavior;
using HRBF.neuralnet.model;
using HRBF.neuralnet.tools;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace HRBF
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            test2();

         
        }

        public void test2()
        {
            var hiddenNeuronCount = 50;
            var inputNeuronCount = 2;

            var data = MeteoData.Read();

            var normalizedData = DataPreparated.NormalizedData(data);
            data = normalizedData.Data;

            var allDataSets = DataPreparated.GetLearningSets(data, inputNeuronCount);


            var learningDataSize = (int)(allDataSets.Count * 0.95);
            var testDataSize = allDataSets.Count - learningDataSize;

            Pair<double[], double>[] learningData = new Pair<double[], double>[learningDataSize];
            Pair<double[], double>[] testData = new Pair<double[], double>[testDataSize];

            Array.Copy(allDataSets.ToArray(), learningData, learningDataSize);
            Array.Copy(allDataSets.ToArray(), learningDataSize, testData, 0, testDataSize);


            var centers = DataPreparated.ExtractCenters(data, inputNeuronCount, hiddenNeuronCount);

            var nn = new HRBFNeuronet(hiddenNeuronCount, inputNeuronCount, centers.Select(t1 => t1.Item1).ToList());


            for (int i = 0; i < 6; i++)
            {
                var l1 = new Pair<double[], double>[31];
                Array.Copy(learningData, i * 31, l1, 0, 31);
                nn.Learning(l1.ToList(), 2500, 0.00002);
            }

           

            var d = learningData.Select(o => o.Item2).ToList();
            var y = testData.Select(o => nn.Calculate(o.Item1)).ToList();



            chart1.Series.Add("y");
            chart1.Series.Add("d");
            
            chart1.Series["y"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            chart1.Series["d"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;

            for (int i = 1; i < y.Count; i++)
            {
                chart1.Series["y"].Points.AddXY(i, y[i]);
                chart1.Series["d"].Points.AddXY(i, d[i]);
            }
        }
    }
}
