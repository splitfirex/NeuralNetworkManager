package io.github.splitfirex.utils;

import java.util.function.Function;

public enum Algorithms {
    SIGMOID, TANH;

    public static Function<Double, Double> function(Algorithms type) {
        switch (type) {
            case SIGMOID:
                return x -> 1.0 / (1.0 + Math.exp(-x));
            case TANH:
                return x -> Math.tanh(x);
            default:
                return null;
        }
    }

    public static Function<Double, Double> derivative(Algorithms type) {
        switch (type) {
            case SIGMOID:
                return x -> x * (1.0 - x);
            case TANH:
                return x -> 1 - Math.pow(Math.tanh(x), 2);
            default:
                return null;
        }
    }

}
