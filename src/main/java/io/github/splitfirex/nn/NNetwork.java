package io.github.splitfirex.nn;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NNetwork {

    private List<Neuron> InputLayer;
    private List<Neuron> OutputLayer;
    private LinkedList<List<Neuron>> HiddenLayers;
    private double LearnRate;
    private double Momentum;
    private Integer idNetwork;

    public NNetwork(int inputSize, int[] hiddenSizes, int outputSize, double learnRate, double momentum) {

        LearnRate = learnRate == 0 ? .4 : learnRate;
        Momentum = momentum == 0 ? .9 : momentum;
        InputLayer = new ArrayList<Neuron>();
        HiddenLayers = new LinkedList<List<Neuron>>();
        OutputLayer = new ArrayList<Neuron>();

        IntStream.range(0, inputSize).forEach(x -> InputLayer.add(new Neuron()));

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


    public void train(List<DataSet> dataSets, int numEpochs) {
        DoubleAdder error = new DoubleAdder();
        AtomicInteger epochCounter = new AtomicInteger(0);

        IntStream.range(0, numEpochs).forEach(x -> {
            List<Double> errors = new ArrayList<>();
            dataSets.stream().forEach(data -> {
                ForwardPropagate(data.input);
                BackPropagate(data.target);
                errors.add(calculateError(data.target));
            });
            error.add(errors.stream().mapToDouble(y -> y).average().getAsDouble());
            //     System.out.println("EPOCH : "+ epochCounter.getAndIncrement() + " E: "+ error.sumThenReset());
        });
    }

    public void train(List<DataSet> dataSets, double minimumError) {
        double error = 1.0;
        int numEpochs = 0;
        AtomicInteger epochCounter = new AtomicInteger(0);

        while (error > minimumError && numEpochs < Integer.MAX_VALUE) {
            List<Double> errors = new ArrayList<>();

            dataSets.forEach(x -> {
                ForwardPropagate(x.input);
                BackPropagate(x.target);
                errors.add(calculateError(x.target));
            });
            error = errors.stream().mapToDouble(x -> x).average().getAsDouble();
            //    System.out.println("EPOCH : "+ epochCounter.getAndIncrement() + " E: "+ error);
            numEpochs++;
        }
    }

    private void ForwardPropagate(double[] inputs) {
        AtomicInteger i = new AtomicInteger();
        InputLayer.stream().forEach(a -> a.setOutput(inputs[i.getAndIncrement()]));
        HiddenLayers.stream().forEach(x -> x.stream().forEach(y -> y.calculateValue()));
        OutputLayer.stream().forEach(a -> a.calculateValue());
    }

    private void BackPropagate(double[] targets) {
        AtomicInteger i = new AtomicInteger();
        OutputLayer.stream().forEach(a -> a.calculateGradient(targets[i.getAndIncrement()]));
        HiddenLayers.stream().collect(Collectors.toCollection(LinkedList::new))
                .descendingIterator().forEachRemaining(x -> x.forEach(y -> y.calculateGradient()));
        HiddenLayers.stream().collect(Collectors.toCollection(LinkedList::new))
                .descendingIterator().forEachRemaining(x -> x.forEach(y -> y.updateWeights(LearnRate, Momentum)));
        OutputLayer.forEach(x -> x.updateWeights(LearnRate, Momentum));

    }

    public double[] compute(double[] inputs) {
        ForwardPropagate(inputs);
        return OutputLayer.stream().mapToDouble(x -> x.output == OutputLayer.stream().mapToDouble(y -> y.getOutput()).max().getAsDouble() ? 1.0 : 0.0).toArray();
    }

    private double calculateError(double[] targets) {
        AtomicInteger i = new AtomicInteger();
        return OutputLayer.stream().mapToDouble(x -> Math.abs(x.calculateError(targets[i.getAndIncrement()]))).sum();
    }

    public Integer getIdNetwork() {
        return idNetwork;
    }

    public void setIdNetwork(Integer idNetwork) {
        this.idNetwork = idNetwork;
    }
}
