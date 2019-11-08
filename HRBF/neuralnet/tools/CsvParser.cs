using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HRBF.neuralnet.tools
{
    public static class CsvParser
    {
        public static List<double> GetColumn(string colName, string path)
        {
            var xList = File.ReadAllLines(path)
                .Select(t1 => t1.Split(','))
                .ToList();
            var headers = xList.First();
            xList.Remove(headers);
            var colIndex = -1;

            for (int i = 0; i < headers.Length; i++)
            {
                if (colName == headers[i]) colIndex = i;
            }
            var xSet = xList.Select(t1 => double.Parse(t1[colIndex].Replace('.', ',')))
                .ToList();
            return xSet;
        }
    }
}
