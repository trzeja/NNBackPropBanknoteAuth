using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBackPropBanknoteAuth
{
    public static class HelperMethods
    {

        public static double[][] ReadDataFromFile(string path)
        {     
            string[] lines = File.ReadAllLines(path);
            
            double[][] result = new double[lines.Length][];            

            for (int i = 0; i < lines.Length; i++)
            {
                result[i] = lines[i].Split(',').Select(lineValue => double.Parse(lineValue)).ToArray();
            }
            //adding 0 or 1 at the end pf vector depending on class, to obtain format 0,1 instead of 0 and 1,0 instead of 1
            int sigleDataVectorLenght = result[0].Length;
            for (int i = 0; i < lines.Length; i++)
            {
                Array.Resize(ref result[i], sigleDataVectorLenght + 1);
                if ((result[i][sigleDataVectorLenght - 1]) == 0.0)
                {
                    result[i][sigleDataVectorLenght] = 1.0;
                }
                else // == 1.0
                {
                    result[i][sigleDataVectorLenght] = 0.0;
                }                
            }

            return result;   
        }
        
        public static double[][] MakeMatrix(int rows, int cols, double initialValue)
        {  
            double[][] result = new double[rows][];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = new double[cols];
                for (int j = 0; j < result[i].Length; j++)
                {
                    result[i][j] = initialValue;
                }
            }

            return result;            
        }

        public static int[] CreateShuffledArray(int size)
        {
            Random rnd = new Random();
            int[] sequence = new int[size];

            for (int i = 0; i < sequence.Length; i++)
            {
                sequence[i] = i;
            }

            for (int i = 0; i < 10; i++) //shuffling
            {
                sequence = sequence.OrderBy(x => rnd.Next()).ToArray();
            }

            return sequence;
        }

        public static void SplitData(double [][] originalData, double trainDataPercentage, out double[][] trainData, out double[][] testData )
        {
            Random rnd = new Random();
            int totalRowsNum = originalData.Length;
            int trainRowsNum = (int)(trainDataPercentage *  totalRowsNum);
            int testRowsNum = totalRowsNum - trainRowsNum;
            trainData = new double[trainRowsNum][];
            testData = new double[testRowsNum][];

            double[][] copyOfOriginalData = new double[totalRowsNum][];
            for (int i = 0; i < copyOfOriginalData.Length; i++) //shallow copy (nothing is going to be modified)
            {
                copyOfOriginalData[i] = originalData[i];
            }

            int[] arrayOfRandomIndexes = CreateShuffledArray(totalRowsNum);

            for (int i = 0, trainRowsCounter = 0, testRowsCounter = 0; i < totalRowsNum; i++)
            {
                int randomIndex = arrayOfRandomIndexes[i];
                if (trainRowsCounter < trainRowsNum)
                {
                    trainData[trainRowsCounter++] = copyOfOriginalData[randomIndex];                    
                }
                else if(testRowsCounter < testRowsNum)
                {
                    testData[testRowsCounter++] = copyOfOriginalData[randomIndex];                    
                }
            }            
        }

        public static double HyperTanh(double x)
        {
            if (x > 20.0)
            {
                return 1.0;
            }
            else if (x < -20.0)
            {
                return -1.0;
            }
            else
            {
                return Math.Tanh(x);
            }           
        }

        public static double[] SoftMax(double [] outputNodesSums)
        {
            double[] result = new double[outputNodesSums.Length];
            double[] nominator = new double[outputNodesSums.Length];
            double denominator = 0.0;

            for (int i = 0; i < nominator.Length; i++)
            {
                denominator += Math.Exp(outputNodesSums[i]); // the same denominator for all nodes
            }

            for (int i = 0; i < nominator.Length; i++)
            {
                nominator[i] = Math.Exp(outputNodesSums[i]);
                result[i] = nominator[i] / denominator;
            }
            return result;
        }       
    }
}
