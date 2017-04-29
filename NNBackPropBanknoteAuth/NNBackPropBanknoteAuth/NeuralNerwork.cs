using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBackPropBanknoteAuth
{
    public class NeuralNerwork
    {
        private int _numInputNodes;
        private int _numHiddenNodes;
        private int _numOutputNodes;

        private double[] _inputs;
        private double[][] _inputToHiddenNodesWeights;
        private double[] _hiddenNodesBiases;
        private double[] _hiddenNodesOutputs;

        private double[][] _hiddenToOutputNodesWeights;
        private double[] _outputNodesBiases;
        private double[] _outputs;

        private string _writeFilePath;

        private Random _rnd; 

        public NeuralNerwork(int numInputNodes, int numHiddenNodes, int numOutputNodes, string writeFilePath)
        {
            _numInputNodes = numInputNodes;
            _numHiddenNodes = numHiddenNodes;
            _numOutputNodes = numOutputNodes;

            _inputs = new double[numInputNodes];
            _inputToHiddenNodesWeights = HelperMethods.MakeMatrix(numInputNodes, numHiddenNodes, 0.0);
            _hiddenNodesBiases = new double[_numHiddenNodes];
            _hiddenNodesOutputs = new double[_numHiddenNodes];

            _hiddenToOutputNodesWeights = HelperMethods.MakeMatrix(numHiddenNodes, numOutputNodes, 0.0);
            _outputNodesBiases = new double[_numOutputNodes];
            _outputs = new double[_numOutputNodes];

            _writeFilePath = writeFilePath;
            _rnd = new Random();
            InitializeWeights();
        }       

        private void InitializeWeights()
        {
            int numAllWeights = (_numInputNodes * _numHiddenNodes) + _numHiddenNodes + (_numHiddenNodes * _numOutputNodes) + _numOutputNodes;
            double[] allWeights = new double[numAllWeights];
            for (int i = 0; i < allWeights.Length; i++)
            {
                allWeights[i] = (0.001 - 0.0001) * _rnd.NextDouble() + 0.0001; //small random values
            }
            SetWeights(allWeights);
        }

        private void SetWeights(double[] newAllSerializedWeights)
        {
            int numAllWeights = (_numInputNodes * _numHiddenNodes)
                + _numHiddenNodes + (_numHiddenNodes * _numOutputNodes) + _numOutputNodes;

            if (numAllWeights != newAllSerializedWeights.Length)
            {
                throw new Exception("Wrong number of weights in SetWeights");
            }
            
            int k = 0;

            for (int i = 0; i < _numInputNodes; i++)
            {
                for (int j = 0; j < _numHiddenNodes; j++)
                {
                    _inputToHiddenNodesWeights[i][j] = newAllSerializedWeights[k++];
                }
            }

            for (int i = 0; i < _numHiddenNodes; i++)
            {
                _hiddenNodesBiases[i] = newAllSerializedWeights[k++];
            }           

            for (int i = 0; i < _numHiddenNodes; i++)
            {
                for (int j = 0; j < _numOutputNodes; j++)
                {
                    _hiddenToOutputNodesWeights[i][j] = newAllSerializedWeights[k++];
                }
            }

            for (int i = 0; i < _numOutputNodes; i++)
            {
                _outputNodesBiases[i] = newAllSerializedWeights[k++];
            }            
        }

        private double[] ComputeOutputs(double[] xValues)
        {            
            double[] hiddenNodesSums = new double[_numHiddenNodes];
            double[] outputNodesSums = new double[_numOutputNodes];

            for (int i = 0; i < _numInputNodes; i++)
            {
                _inputs[i] = xValues[i];
            }

            for (int j = 0; j < _numHiddenNodes; j++)
            {
                for (int i = 0; i < _numInputNodes; i++)
                {
                    hiddenNodesSums[j] += _inputs[i] * _inputToHiddenNodesWeights[i][j];
                }
            }

            for (int i = 0; i < _numHiddenNodes; i++)
            {
                hiddenNodesSums[i] += _hiddenNodesBiases[i];
            }

            for (int i = 0; i < _numHiddenNodes; i++)
            {
                _hiddenNodesOutputs[i] = HelperMethods.HyperTanh(hiddenNodesSums[i]);
            }

            for (int j = 0; j < _numOutputNodes; j++)
            {
                for (int i = 0; i < _numHiddenNodes; i++)
                {
                    outputNodesSums[j] += _hiddenNodesOutputs[i] * _hiddenToOutputNodesWeights[i][j];
                }
            }

            for (int i = 0; i < _numOutputNodes; i++)
            {
                outputNodesSums[i] += _outputNodesBiases[i];
            }

            double[] softMaxedOutputs = HelperMethods.SoftMax(outputNodesSums);
            Array.Copy(softMaxedOutputs, _outputs, softMaxedOutputs.Length);

            double[] result = new double[_numOutputNodes];
            Array.Copy(_outputs, result, _outputs.Length);
            return result;
        }

        public void Train(double[][] trainData, double[][] testData, int maxEpochs, double learnRate, double momentum, double certainty)
        {
            double[][] hiddenToOutputNodesGradients = HelperMethods.MakeMatrix(_numHiddenNodes, _numOutputNodes, 0.0);
            double[] outputNodesBiasGradients = new double[_numOutputNodes];
            double[][] inputToHiddenNodesGradients = HelperMethods.MakeMatrix(_numInputNodes, _numHiddenNodes, 0.0);
            double[] hiddenNodesBiasGradients = new double[_numHiddenNodes];

            double[] outputLocalErrorGradientSignals = new double[_numOutputNodes];
            double[] hiddenLocalErrorGradientSignals = new double[_numHiddenNodes];

            double[][] inputToHiddenPrevWeightsDelta = HelperMethods.MakeMatrix(_numInputNodes, _numHiddenNodes, 0.0);
            double[] hiddenPrevBiasesDelta = new double[_numHiddenNodes];
            double[][] hiddenToOutputPrevWeightsDelta = HelperMethods.MakeMatrix(_numHiddenNodes, _numOutputNodes, 0.0);
            double[] outputPrevBiasesDelta = new double[_numOutputNodes];

            int epoch = 0;
            double[] inputValues = new double[_numInputNodes];
            double[] targetOutputs = new double[_numOutputNodes];
            double derivative = 0.0;
            double errorSignal = 0.0;            
            double errorInterval = maxEpochs / 100;

            int[] sequence = HelperMethods.CreateShuffledArray(trainData.Length); // in result shuffling is made only once

            StringBuilder outputText = new StringBuilder();
            //outputText.Append("Hidden nodes: " + _numHiddenNodes + " epochs: "
            //    + maxEpochs + " learn rate: " + learnRate + " momentum " 
            //    + momentum.ToString("F3") + Environment.NewLine);
            outputText.Append("epoch: MSE: TrainAcc: TestAcc:" + Environment.NewLine);


            while (epoch < maxEpochs)
            {
                //epoch++; // for diagrams     
                if (epoch % errorInterval == 0 && epoch < maxEpochs)                
                {
                    double trainDataAccuracy = Accuracy(trainData,certainty);
                    double testDataAccuracy = Accuracy(testData, certainty);

                    double error = CalculateMeanSquaredError(trainData);
                    Console.WriteLine("epoch: " + epoch + "\t MSE: \t" + error.ToString("F10") +
                       "\tTrain Acc: \t" + trainDataAccuracy + "\t Test Acc: \t" + testDataAccuracy);
                    outputText.Append(epoch.ToString() + 
                        " " + error.ToString("F15").Replace(".",",") + //write to file (commas separated)
                        " " + trainDataAccuracy.ToString("F15").Replace(".", ",") + 
                        " " + testDataAccuracy.ToString("F15").Replace(".", ",") + Environment.NewLine); 
                }
                epoch++;

                for (int ii = 0; ii < trainData.Length; ii++)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], inputValues, _numInputNodes);
                    Array.Copy(trainData[idx], _numInputNodes, targetOutputs, 0, _numOutputNodes);
                    ComputeOutputs(inputValues);

                    //i - inputs, j - hidden, k - outputs

                    for (int k = 0; k < _numOutputNodes; k++)
                    {   
                        errorSignal =  targetOutputs[k] - _outputs[k];
                        derivative = _outputs[k] * (1.0 - _outputs[k]); //using softMax
                        outputLocalErrorGradientSignals[k] = errorSignal * derivative;
                    }

                    for (int j = 0; j < _numHiddenNodes; j++)
                    {
                        for (int k = 0; k < _numOutputNodes; k++)
                        {
                            hiddenToOutputNodesGradients[j][k] = outputLocalErrorGradientSignals[k] * _hiddenNodesOutputs[j];
                        }
                    }

                    for (int k = 0; k < _numOutputNodes; k++)
                    {
                        outputNodesBiasGradients[k] = outputLocalErrorGradientSignals[k] * 1.0; 
                    }

                    for (int j = 0; j < _numHiddenNodes; j++)
                    {
                        derivative = (1.0 - _hiddenNodesOutputs[j]) * (1.0 + _hiddenNodesOutputs[j]); //using tanh
                        double sum = 0.0;
                        for (int k = 0; k < _numOutputNodes; k++)
                        {
                            sum += outputLocalErrorGradientSignals[k] * _hiddenToOutputNodesWeights[j][k];
                        }

                        hiddenLocalErrorGradientSignals[j] = sum * derivative;
                    }

                    for (int i = 0; i < _numInputNodes; i++)
                    {
                        for (int j = 0; j < _numHiddenNodes; j++)
                        {
                            inputToHiddenNodesGradients[i][j] = hiddenLocalErrorGradientSignals[j] * _inputs[i];
                        }
                    }

                    for (int j = 0; j < _numHiddenNodes; j++)
                    {
                        hiddenNodesBiasGradients[j] = hiddenLocalErrorGradientSignals[j] * 1.0;
                    }

                    //  update weights and biases

                    for (int i = 0; i < _numInputNodes; i++)
                    {
                        for (int j = 0; j < _numHiddenNodes; j++)
                        {
                            double delta = learnRate * inputToHiddenNodesGradients[i][j];
                            _inputToHiddenNodesWeights[i][j] += delta;
                            _inputToHiddenNodesWeights[i][j] += inputToHiddenPrevWeightsDelta[i][j] * momentum;
                            inputToHiddenPrevWeightsDelta[i][j] = delta; //save
                        }
                    }

                    for (int j = 0; j < _numHiddenNodes; j++)
                    {
                        double delta = learnRate * hiddenNodesBiasGradients[j];
                        _hiddenNodesBiases[j] += delta;
                        _hiddenNodesBiases[j] += hiddenPrevBiasesDelta[j] * momentum;
                        hiddenPrevBiasesDelta[j] = delta; //save
                    }

                    for (int j = 0; j < _numHiddenNodes; j++)
                    {
                        for (int k = 0; k < _numOutputNodes; k++)
                        {
                            double delta = learnRate * hiddenToOutputNodesGradients[j][k];
                            _hiddenToOutputNodesWeights[j][k] += delta;
                            _hiddenToOutputNodesWeights[j][k] += hiddenToOutputPrevWeightsDelta[j][k] * momentum;
                            hiddenToOutputPrevWeightsDelta[j][k] = delta; //save
                        }
                    }

                    for (int k = 0; k < _numOutputNodes; k++)
                    {
                        double delta = learnRate * outputNodesBiasGradients[k];
                        _outputNodesBiases[k] += delta;
                        _outputNodesBiases[k] += outputPrevBiasesDelta[k] * momentum;
                        outputPrevBiasesDelta[k] = delta; //save
                    }
                }// each data vector                
            } // while epochs
            File.WriteAllText(_writeFilePath, outputText.ToString());            
        } // train method

        private double CalculateMeanSquaredError(double[][] trainData)
        {
            double sumOfErrors = 0.0;
            double[] inputValues = new double[_numInputNodes];
            double[] targetOutputs = new double[_numOutputNodes];

            for (int i = 0; i < trainData.Length; i++) //for every data vector
            {
                Array.Copy(trainData[i], inputValues, _numInputNodes);
                Array.Copy(trainData[i], _numInputNodes, targetOutputs, 0, _numOutputNodes);
                double[] computedOutputs = ComputeOutputs(inputValues);
                for (int j = 0; j < _numOutputNodes; j++)
                {
                    double error = targetOutputs[j] - computedOutputs[j];
                    sumOfErrors += error * error;
                }
            }
                                    
            return sumOfErrors / trainData.Length;
        }

        private double Accuracy(double[][] data, double certainty)
        {
            int Correct = 0;
            int Incorrect = 0;
            int targetOutputsMaxValueIndex;
            int computedOutputsMaxValueIndex;
            double[] inputValues = new double[_numInputNodes];
            double[] targetOutputs = new double[_numOutputNodes];

            for (int i = 0; i < data.Length; i++) //for every data vector
            {                
                Array.Copy(data[i], inputValues, _numInputNodes);
                Array.Copy(data[i], _numInputNodes, targetOutputs, 0, _numOutputNodes);
                double[] computedOutputs = ComputeOutputs(inputValues);

                targetOutputsMaxValueIndex = GetMaxValueIndex(targetOutputs);
                computedOutputsMaxValueIndex = GetMaxValueIndex(computedOutputs);
                
                if (Math.Abs(computedOutputs[0] - computedOutputs[1]) < certainty) //if network is unsure
                {
                    Incorrect++;
                    continue;
                }

                if (targetOutputsMaxValueIndex == computedOutputsMaxValueIndex)
                {
                    Correct++;
                }
                else
                {
                    Incorrect++;
                }
            }
            double result = ((double)Correct) / (Correct + Incorrect);
            return result;
        }

        private int GetMaxValueIndex(double[] vector)
        {
            int maxIndex = 0;
            double maxValue = vector[0];
            for (int i = 0; i < vector.Length; i++)
            {
                if (maxValue < vector[i])
                {
                    maxIndex = i;
                    maxValue = vector[i];
                }
            }
            return maxIndex;
        }
    }
}
