using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HRBF.neuralnet.tools
{
    public static class MathHelper
    {
        private const double COEF = 1.0 / 2.0;
        public static Random Rnd { get; set; } = new Random();


        public static double CalculateError(double y, double d) =>
            COEF * Math.Pow(y - d, 2);
    }
}
