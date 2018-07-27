package io.github.splitfirex.nn;

import io.github.splitfirex.utils.PerformanceHelper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.Executors;
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
    private ConcurrentLinkedDeque<Double> errorsGlobal = new ConcurrentLinkedDeque<>();

    public ConcurrentLinkedDeque<Double> getErrorsGlobal() {
        return errorsGlobal;
    }

    public void setErrorsGlobal(ConcurrentLinkedDeque<Double> errorsGlobal) {
        this.errorsGlobal = errorsGlobal;
    }

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


    public void train(List<DataSet> dataSets, int percentageTest, int numEpochs) {
        DoubleAdder error = new DoubleAdder();
        AtomicInteger epochCounter = new AtomicInteger(0);

        Collections.shuffle(dataSets);
        List<DataSet> trainData = dataSets.subList(0, (dataSets.size() * (100 - percentageTest)) / 100);
        List<DataSet> testData = dataSets.subList((100 - percentageTest) / 100, dataSets.size() - 1);

        IntStream.range(0, numEpochs).forEach(x -> {
            List<Double> errors = new ArrayList<>();
            trainData.stream().forEach(data -> {
                ForwardPropagate(data.input);
                BackPropagate(data.target);
                errors.add(calculateError(data.target));
            });
            error.add(errors.stream().mapToDouble(y -> y).average().getAsDouble());
            errorsGlobal.add(error.sum());
            System.out.println("EPOCH : " + epochCounter.getAndIncrement() + " E: " + error.sumThenReset());
        });

        List accum = new ArrayList<>();
        testData.stream().forEach(y -> PerformanceHelper.efficency(accum, this.compute(y.getInput()), y.getTarget()));
        PerformanceHelper.total(accum);
    }

    public void train(List<DataSet> dataSets, int percentageTest) {
        Executors.newCachedThreadPool().submit(() -> {
            double error = 1.0;
            int numEpochs = 0;
            AtomicInteger epochCounter = new AtomicInteger(0);

            long startTime = System.currentTimeMillis();

            Collections.shuffle(dataSets);
            List<DataSet> trainData = dataSets.subList(0, (int) ((double) dataSets.size() * ((double) (100 - percentageTest) / (double) 100)));
            List<DataSet> testData = dataSets.subList((int) ((double) dataSets.size() * ((double) (100 - percentageTest) / (double) 100)), dataSets.size() - 1);

            Integer count = 0;
            double lastError = 0.0;

            while (numEpochs < Integer.MAX_VALUE && count <= 5) {
                List<Double> errors = new ArrayList<>();

                trainData.forEach(x -> {
                    ForwardPropagate(x.input);
                    BackPropagate(x.target);
                    errors.add(calculateError(x.target));
                });
                error = errors.stream().mapToDouble(x -> x).average().getAsDouble();
                errorsGlobal.add(error);
                //      System.out.println("EPOCH : " + epochCounter.getAndIncrement() + " E: " + (double) (Math.round((double) error * 100000.0) / 100000.0));
                if (lastError == (double) Math.round(error * 100000.0) / 100000.0) {
                    count++;
                } else {
                    lastError = Math.round(error * 100000.0) / 100000.0;
                    count = 0;
                }
                numEpochs++;
            }

            List accum = new ArrayList<>();
            testData.stream().forEach(y -> PerformanceHelper.efficency(accum, this.compute(y.getInput()), y.getTarget()));
            PerformanceHelper.total(accum);

            long stopTime = System.currentTimeMillis();
            long elapsedTime = stopTime - startTime;
            System.out.println("Tiempo:" + elapsedTime / 1000.0);
            return null;
        });
    }

    public void train(List<DataSet> dataSets, int percentageTest, double minimumError) {
        double error = 1.0;
        int numEpochs = 0;
        AtomicInteger epochCounter = new AtomicInteger(0);

        Collections.shuffle(dataSets);
        List<DataSet> trainData = dataSets.subList(0, (int) ((double) dataSets.size() * ((double) (100 - percentageTest) / (double) 100)));
        List<DataSet> testData = dataSets.subList((int) ((double) dataSets.size() * ((double) (100 - percentageTest) / (double) 100)), dataSets.size() - 1);

        while (error > minimumError && numEpochs < Integer.MAX_VALUE) {
            List<Double> errors = new ArrayList<>();

            trainData.forEach(x -> {
                ForwardPropagate(x.input);
                BackPropagate(x.target);
                errors.add(calculateError(x.target));
            });
            error = errors.stream().mapToDouble(x -> x).average().getAsDouble();
            errorsGlobal.add(error);
            System.out.println("EPOCH : " + epochCounter.getAndIncrement() + " E: " + error);
            numEpochs++;
        }

        List accum = new ArrayList<>();
        testData.stream().forEach(y -> PerformanceHelper.efficency(accum, this.compute(y.getInput()), y.getTarget()));
        PerformanceHelper.total(accum);
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
