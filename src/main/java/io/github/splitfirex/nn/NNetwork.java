package io.github.splitfirex.nn;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NNetwork {

    private List<Neuron> InputLayer;
    private List<Neuron> OutputLayer;
    private LinkedList<List<Neuron>> HiddenLayers;
    private double LearnRate;
    private double Momentum;

    public NNetwork(int inputSize, int[] hiddenSizes, int outputSize, double learnRate, double momentum) {

        LearnRate = learnRate == 0 ? .4 : learnRate;
        Momentum = momentum == 0 ? .9 : momentum;
        InputLayer = new ArrayList<Neuron>();
        HiddenLayers = new LinkedList<>();
        OutputLayer = new ArrayList<Neuron>();

        List<Neuron> firstHiddenLayer = new ArrayList<>();
        IntStream.range(0, hiddenSizes[0]).forEach(x -> firstHiddenLayer.add(new Neuron(InputLayer)));
        HiddenLayers.add(firstHiddenLayer);

        IntStream.range(1, hiddenSizes.length).forEach(x -> {
            List<Neuron> hiddenLayer = new ArrayList<>();
            IntStream.range(0, hiddenSizes[x]).forEach(y -> hiddenLayer.add(new Neuron(HiddenLayers.get(x - 1))));
            HiddenLayers.add(hiddenLayer);
        });

        IntStream.range(0, outputSize).forEach(x -> OutputLayer.add(new Neuron(HiddenLayers.getLast())));
    }


    public void Train(List<DataSet> dataSets, int numEpochs) {
        IntStream.range(0, numEpochs).forEach(x -> {

            dataSets.stream().forEach(data -> {
                ForwardPropagate(data.input);
                BackPropagate(data.target);

            });
        });
    }

    public void Train(List<DataSet> dataSets, double minimumError) {
        double error = 1.0;
        int numEpochs = 0;

        while (error > minimumError && numEpochs < Integer.MAX_VALUE) {
            List<Double> errors = new ArrayList<>();

            dataSets.forEach(x -> {
                ForwardPropagate(x.input);
                BackPropagate(x.target);
                errors.add(CalculateError(x.target));

            });
            error = errors.stream().mapToDouble(x -> x).average().getAsDouble();
            numEpochs++;
        }
    }

    private void ForwardPropagate(double[] inputs) {
        AtomicInteger i = new AtomicInteger();
        InputLayer.stream().forEach(a -> a.setOutput(inputs[i.getAndIncrement()]));
        HiddenLayers.stream().forEach(x -> x.stream().forEach(y -> y.CalculateValue()));
        OutputLayer.stream().forEach(a -> a.CalculateValue());
    }

    private void BackPropagate(double[] targets) {
        AtomicInteger i = new AtomicInteger();
        OutputLayer.stream().forEach(a -> a.CalculateGradient(targets[i.getAndIncrement()]));
        HiddenLayers.stream().collect(Collectors.toCollection(LinkedList::new))
                .descendingIterator().forEachRemaining(x -> x.forEach(y -> y.CalculateGradient()));
        HiddenLayers.stream().collect(Collectors.toCollection(LinkedList::new))
                .descendingIterator().forEachRemaining(x -> x.forEach(y -> y.UpdateWeights(LearnRate, Momentum)));
        OutputLayer.forEach(x -> x.UpdateWeights(LearnRate, Momentum));

    }

    public double[] Compute(double[] inputs) {
        ForwardPropagate(inputs);
        return OutputLayer.stream().mapToDouble(x -> x.getOutput()).toArray();
    }

    private double CalculateError(double[] targets) {
        AtomicInteger i = new AtomicInteger();
        return OutputLayer.stream().mapToDouble(x -> Math.abs(x.CalculateError(targets[i.getAndIncrement()]))).sum();
    }
}
