package io.github.splitfirex.nn;

import io.github.splitfirex.utils.GenWeigths;

public class Axiom {

    Neuron inputNeuron;
    Neuron outputNeuron;

    double weight;
    double deltaWeight;


    public Axiom(Neuron inputNeuron, Neuron outputNeuron) {
        inputNeuron = inputNeuron;
        outputNeuron = outputNeuron;
        weight = (GenWeigths.generator.nextDouble() * 2) - 1;
    }

}
