package io.github.splitfirex.nn;

public class DataSet {

    double[] input;
    double[] target;

    public DataSet(double[] input, double[] target) {
        this.target = target;
        this.input = input;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getTarget() {
        return target;
    }

    public void setTarget(double[] target) {
        this.target = target;
    }
}
