package io.github.splitfirex.utils;

import java.util.function.Function;

public enum Algorithms {
    SIGMOID;

    public static Function<Double, Double> function(Algorithms type) {
        switch (type) {
            case SIGMOID:
                return x -> x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Math.exp(-x));
            default:
                return null;
        }
    }

    public static Function<Double, Double> derivative(Algorithms type) {
        switch (type) {
            case SIGMOID:
                return x -> x * (x - 1.0);

            default:
                return null;
        }
    }

}
