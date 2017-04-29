using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NNBackPropBanknoteAuth
{
    class MainProgram
    {
        static void Main(string[] args)
        {
            //numInputNodes = 4;
            //numHiddenNodes = 5;
            //numOutputNodes = 2;
            //trainDataPercentage = 0.8;
            //maxEpochs = 1000;
            //learnRate = 0.05;
            //momentum = 0.01;
            //certainty = 0.99;

            //defaults
            int numInputNodes = 4;
            int numHiddenNodes = 5;
            int numOutputNodes = 2;
            double trainDataPercentage = 0.8;
            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.05;
            double certainty = 0.99;
            string dataSetFullPath = @"C:\Users\trzej_000\Google Drive\Politechniczne\BIAI\projekt\NNBackPropBanknoteAuth\BanknoteAuthenticationData.txt";
            string writeFileFullPath = @"C:\Users\trzej_000\Google Drive\Politechniczne\BIAI\projekt\NNBackPropBanknoteAuth\BanknoteAuthenticationOutput.txt";

            if (args.Length == 2) //data set path given by the user
            {               
                dataSetFullPath = args[1];
                writeFileFullPath = dataSetFullPath.Remove(dataSetFullPath.LastIndexOf("\\"));
                writeFileFullPath += @"\BanknoteAuthenticationOutput.txt";
            }
            else if (args.Length == 11) //all parameters given by the user
            {
                dataSetFullPath = args[1];
                writeFileFullPath = args[2];
                numInputNodes = Int32.Parse(args[3]);
                numHiddenNodes = Int32.Parse(args[4]);
                numOutputNodes = Int32.Parse(args[5]);
                trainDataPercentage = double.Parse(args[6]);
                maxEpochs = Int32.Parse(args[7]);
                learnRate = double.Parse(args[8]);
                momentum = double.Parse(args[9]);
                certainty = double.Parse(args[10]);
            }

            double[][] trainData;
            double[][] testData;           

            Console.WriteLine("Starting program");
            double[][] originalData =  HelperMethods.ReadDataFromFile(dataSetFullPath);
            HelperMethods.SplitData(originalData, trainDataPercentage, out trainData, out testData);            

            NeuralNerwork nn = new NeuralNerwork(numInputNodes, numHiddenNodes, numOutputNodes, writeFileFullPath);
            nn.Train(trainData, testData, maxEpochs, learnRate,momentum,certainty);

            Console.WriteLine("End of program");
            Console.ReadKey();
        } // Main
    }
}
