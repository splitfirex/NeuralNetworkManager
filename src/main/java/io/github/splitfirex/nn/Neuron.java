package io.github.splitfirex.nn;

import io.github.splitfirex.utils.Algorithms;
import io.github.splitfirex.utils.GenWeigths;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    Algorithms activationFunction = Algorithms.SIGMOID;

    double bias;
    double output;
    double delta;
    double gradient;

    List<Axiom> inputAxioms;
    List<Axiom> outputAxioms;

    public Neuron() {
        inputAxioms = new ArrayList<Axiom>();
        outputAxioms = new ArrayList<Axiom>();
        bias = (GenWeigths.generator.nextDouble() * 2) - 1;
    }


    public Neuron(List<Neuron> inputNeurons) {
        this();
        inputNeurons.stream().forEach(neuron -> {
            Axiom axiom = new Axiom(neuron, this);
            neuron.outputAxioms.add(axiom);
            inputAxioms.add(axiom);
        });
    }

    public double calculateValue() {
        output = Algorithms.function(activationFunction)
                .apply(inputAxioms.stream().map(x -> x.weight * x.inputNeuron.getOutput()).mapToDouble(x -> x).sum() + bias);
        return output;
    }

    public double calculateError(double target) {
        return target - output;
    }

    public double calculateGradient(double target) {
        gradient = calculateError(target) * Algorithms.derivative(activationFunction).apply(output);
        return gradient;
    }

    public double calculateGradient() {
        gradient = outputAxioms.stream().mapToDouble(x -> x.outputNeuron.gradient * x.weight).sum() * Algorithms.derivative(activationFunction).apply(output);
        return gradient;
    }

    public void updateWeights(double learnRate, double momentum) {
        double prevDelta = delta;
        delta = learnRate * gradient;
        bias += delta + (momentum * prevDelta);

        inputAxioms.stream().forEach(x -> {
            double prevD = x.deltaWeight;
            x.deltaWeight = learnRate * gradient * x.inputNeuron.getOutput();
            x.weight += x.deltaWeight + (momentum * prevD);
        });
    }


    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getGradient() {
        return gradient;
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }
}
